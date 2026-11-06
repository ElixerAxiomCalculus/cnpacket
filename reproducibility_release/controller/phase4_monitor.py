import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls
)
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib.packet import ipv4
from ryu.lib.packet import ipv6
from ryu.lib.packet import packet
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.ofproto import ofproto_v1_3


class ResearchPhase4Switch(
    app_manager.RyuApp
):
    OFP_VERSIONS = [
        ofproto_v1_3.OFP_VERSION
    ]

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.mac_to_port = {}
        self.flow_states = {}

        self.mode = os.environ.get(
            "PHASE4_MODE",
            "forwarding"
        )

        if self.mode not in {
            "forwarding",
            "features",
            "inference"
        }:
            raise RuntimeError(
                f"Invalid PHASE4_MODE={self.mode}"
            )

        self.telemetry_dpid = int(
            os.environ.get(
                "PHASE4_TELEMETRY_DPID",
                "1"
            )
        )

        self.sequence_packets = int(
            os.environ.get(
                "PHASE4_SEQUENCE_PACKETS",
                "5"
            )
        )

        self.flow_idle_seconds = float(
            os.environ.get(
                "PHASE4_FLOW_IDLE_SECONDS",
                "120"
            )
        )

        self.inference_url = os.environ.get(
            "PHASE4_INFERENCE_URL",
            "http://127.0.0.1:8765/infer"
        )

        self.metrics_path = Path(
            os.environ.get(
                "PHASE4_METRICS_PATH",
                "artifacts/phase4/controller_metrics.jsonl"
            )
        )

        self.metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.telemetry_packets = 0

        self.logger.info(
            "PHASE4_CONTROLLER_START "
            "mode=%s telemetry_dpid=%s "
            "sequence_packets=%s",
            self.mode,
            self.telemetry_dpid,
            self.sequence_packets
        )

    def add_flow(
        self,
        datapath,
        priority,
        match,
        actions,
        buffer_id=None,
        idle_timeout=0,
        hard_timeout=0
    ):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        instructions = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]

        arguments = {
            "datapath": datapath,
            "priority": priority,
            "match": match,
            "instructions": instructions,
            "idle_timeout": idle_timeout,
            "hard_timeout": hard_timeout
        }

        if buffer_id is not None:
            arguments[
                "buffer_id"
            ] = buffer_id

        datapath.send_msg(
            parser.OFPFlowMod(
                **arguments
            )
        )

    @set_ev_cls(
        ofp_event.EventOFPSwitchFeatures,
        CONFIG_DISPATCHER
    )
    def switch_features_handler(
        self,
        ev
    ):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id

        self.mac_to_port.setdefault(
            dpid,
            {}
        )

        match = parser.OFPMatch()

        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]

        self.add_flow(
            datapath,
            0,
            match,
            actions
        )

        self.logger.info(
            "SWITCH_CONNECTED dpid=%016x",
            dpid
        )

    def _flow_information(
        self,
        pkt
    ):
        ipv4_packet = pkt.get_protocol(
            ipv4.ipv4
        )

        ipv6_packet = pkt.get_protocol(
            ipv6.ipv6
        )

        tcp_packet = pkt.get_protocol(
            tcp.tcp
        )

        udp_packet = pkt.get_protocol(
            udp.udp
        )

        if (
            ipv4_packet is None
            and ipv6_packet is None
        ):
            return None

        if (
            tcp_packet is None
            and udp_packet is None
        ):
            return None

        if ipv4_packet is not None:
            ip_version = 4
            source_ip = ipv4_packet.src
            destination_ip = ipv4_packet.dst
        else:
            ip_version = 6
            source_ip = ipv6_packet.src
            destination_ip = ipv6_packet.dst

        if tcp_packet is not None:
            protocol = 6
            source_port = int(
                tcp_packet.src_port
            )
            destination_port = int(
                tcp_packet.dst_port
            )
        else:
            protocol = 17
            source_port = int(
                udp_packet.src_port
            )
            destination_port = int(
                udp_packet.dst_port
            )

        source_endpoint = (
            source_ip,
            source_port
        )

        destination_endpoint = (
            destination_ip,
            destination_port
        )

        ordered = sorted(
            (
                source_endpoint,
                destination_endpoint
            )
        )

        key = (
            ip_version,
            protocol,
            ordered[
                0
            ][
                0
            ],
            ordered[
                0
            ][
                1
            ],
            ordered[
                1
            ][
                0
            ],
            ordered[
                1
            ][
                1
            ]
        )

        return {
            "key": key,
            "ip_version": ip_version,
            "protocol": protocol,
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "source_port": source_port,
            "destination_port": destination_port,
            "source_endpoint": source_endpoint,
            "destination_endpoint": destination_endpoint
        }

    def _flow_id(
        self,
        key
    ):
        return hashlib.sha256(
            repr(
                key
            ).encode(
                "utf-8"
            )
        ).hexdigest()[
            :24
        ]

    def _clean_states(
        self,
        now_ns
    ):
        threshold_ns = int(
            self.flow_idle_seconds
            * 1_000_000_000
        )

        stale = [
            key
            for key, state in self.flow_states.items()
            if (
                now_ns
                - state[
                    "last_seen_ns"
                ]
            )
            > threshold_ns
        ]

        for key in stale:
            del self.flow_states[
                key
            ]

    def _infer(
        self,
        signed_sizes,
        iats
    ):
        payload = json.dumps(
            {
                "signed_packet_sizes": signed_sizes,
                "interarrival_seconds": iats
            },
            separators=(
                ",",
                ":"
            )
        ).encode(
            "utf-8"
        )

        request = urllib.request.Request(
            self.inference_url,
            data=payload,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        start_ns = time.perf_counter_ns()

        with urllib.request.urlopen(
            request,
            timeout=2.0
        ) as response:
            body = response.read()

        end_ns = time.perf_counter_ns()

        result = json.loads(
            body.decode(
                "utf-8"
            )
        )

        return (
            result,
            (
                end_ns
                - start_ns
            )
            / 1000.0
        )

    def _record(
        self,
        payload
    ):
        with self.metrics_path.open(
            "a",
            encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    payload,
                    separators=(
                        ",",
                        ":"
                    )
                )
                + "\n"
            )

    def _process_telemetry(
        self,
        msg,
        pkt,
        frame_length,
        packet_in_start_ns
    ):
        feature_start_ns = time.perf_counter_ns()

        information = self._flow_information(
            pkt
        )

        if information is None:
            return None

        now_ns = time.monotonic_ns()

        self.telemetry_packets += 1

        if (
            self.telemetry_packets
            % 1000
        ) == 0:
            self._clean_states(
                now_ns
            )

        key = information[
            "key"
        ]

        state = self.flow_states.get(
            key
        )

        if state is None:
            state = {
                "first_source": information[
                    "source_endpoint"
                ],
                "first_destination": information[
                    "destination_endpoint"
                ],
                "last_seen_ns": now_ns,
                "signed_sizes": [],
                "iats": [],
                "feature_total_us": 0.0,
                "complete": False
            }

            self.flow_states[
                key
            ] = state

        if state[
            "complete"
        ]:
            state[
                "last_seen_ns"
            ] = now_ns

            return {
                "information": information,
                "complete": True,
                "complete_now": False
            }

        if (
            information[
                "source_endpoint"
            ]
            == state[
                "first_source"
            ]
            and information[
                "destination_endpoint"
            ]
            == state[
                "first_destination"
            ]
        ):
            direction = 1
        else:
            direction = -1

        if not state[
            "signed_sizes"
        ]:
            iat = 0.0
        else:
            iat = (
                now_ns
                - state[
                    "last_seen_ns"
                ]
            ) / 1_000_000_000.0

            if iat < 0:
                iat = 0.0

        state[
            "signed_sizes"
        ].append(
            int(
                direction
                * frame_length
            )
        )

        state[
            "iats"
        ].append(
            float(
                iat
            )
        )

        state[
            "last_seen_ns"
        ] = now_ns

        complete_now = (
            len(
                state[
                    "signed_sizes"
                ]
            )
            >= self.sequence_packets
        )

        feature_end_ns = time.perf_counter_ns()

        feature_packet_us = (
            feature_end_ns
            - feature_start_ns
        ) / 1000.0

        state[
            "feature_total_us"
        ] += feature_packet_us

        if not complete_now:
            return {
                "information": information,
                "complete": False,
                "complete_now": False
            }

        state[
            "complete"
        ] = True

        record = {
            "mode": self.mode,
            "flow_id": self._flow_id(
                key
            ),
            "protocol": information[
                "protocol"
            ],
            "sequence_packets": len(
                state[
                    "signed_sizes"
                ]
            ),
            "feature_last_packet_us": feature_packet_us,
            "feature_total_us": state[
                "feature_total_us"
            ]
        }

        if self.mode == "inference":
            try:
                (
                    inference,
                    inference_roundtrip_us
                ) = self._infer(
                    state[
                        "signed_sizes"
                    ][
                        :self.sequence_packets
                    ],
                    state[
                        "iats"
                    ][
                        :self.sequence_packets
                    ]
                )

                record[
                    "inference_roundtrip_us"
                ] = inference_roundtrip_us

                record[
                    "service_us"
                ] = inference[
                    "service_us"
                ]

                record[
                    "model_us"
                ] = inference[
                    "model_us"
                ]

                record[
                    "prediction"
                ] = inference[
                    "prediction"
                ]

                record[
                    "nearest_known_family"
                ] = inference[
                    "nearest_known_family"
                ]

                record[
                    "unknown_score"
                ] = inference[
                    "unknown_score"
                ]

                record[
                    "threshold"
                ] = inference[
                    "threshold"
                ]

            except Exception as exc:
                record[
                    "inference_error"
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        record[
            "controller_decision_us"
        ] = (
            time.perf_counter_ns()
            - packet_in_start_ns
        ) / 1000.0

        self._record(
            record
        )

        self.logger.info(
            "PHASE4_FLOW_COMPLETE "
            "mode=%s flow_id=%s packets=%s "
            "feature_total_us=%.3f decision_us=%.3f",
            self.mode,
            record[
                "flow_id"
            ],
            record[
                "sequence_packets"
            ],
            record[
                "feature_total_us"
            ],
            record[
                "controller_decision_us"
            ]
        )

        return {
            "information": information,
            "complete": True,
            "complete_now": True
        }

    def _telemetry_actions(
        self,
        datapath,
        out_port
    ):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        actions = [
            parser.OFPActionOutput(
                out_port
            )
        ]

        if (
            self.mode
            in {
                "features",
                "inference"
            }
            and datapath.id
            == self.telemetry_dpid
        ):
            actions.append(
                parser.OFPActionOutput(
                    ofproto.OFPP_CONTROLLER,
                    ofproto.OFPCML_NO_BUFFER
                )
            )

        return actions

    def _install_bypass(
        self,
        datapath,
        in_port,
        eth,
        information,
        out_port
    ):
        if out_port == datapath.ofproto.OFPP_FLOOD:
            return

        parser = datapath.ofproto_parser

        arguments = {
            "in_port": in_port,
            "eth_src": eth.src,
            "eth_dst": eth.dst,
            "ip_proto": information[
                "protocol"
            ]
        }

        if information[
            "ip_version"
        ] == 4:
            arguments[
                "eth_type"
            ] = ether_types.ETH_TYPE_IP

            arguments[
                "ipv4_src"
            ] = information[
                "source_ip"
            ]

            arguments[
                "ipv4_dst"
            ] = information[
                "destination_ip"
            ]

        else:
            arguments[
                "eth_type"
            ] = ether_types.ETH_TYPE_IPV6

            arguments[
                "ipv6_src"
            ] = information[
                "source_ip"
            ]

            arguments[
                "ipv6_dst"
            ] = information[
                "destination_ip"
            ]

        if information[
            "protocol"
        ] == 6:
            arguments[
                "tcp_src"
            ] = information[
                "source_port"
            ]

            arguments[
                "tcp_dst"
            ] = information[
                "destination_port"
            ]

        else:
            arguments[
                "udp_src"
            ] = information[
                "source_port"
            ]

            arguments[
                "udp_dst"
            ] = information[
                "destination_port"
            ]

        match = parser.OFPMatch(
            **arguments
        )

        actions = [
            parser.OFPActionOutput(
                out_port
            )
        ]

        self.add_flow(
            datapath,
            20,
            match,
            actions,
            idle_timeout=int(
                self.flow_idle_seconds
            )
        )

    @set_ev_cls(
        ofp_event.EventOFPPacketIn,
        MAIN_DISPATCHER
    )
    def packet_in_handler(
        self,
        ev
    ):
        packet_in_start_ns = time.perf_counter_ns()

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        dpid = datapath.id
        in_port = msg.match[
            "in_port"
        ]

        pkt = packet.Packet(
            msg.data
        )

        eth = pkt.get_protocol(
            ethernet.ethernet
        )

        if eth is None:
            return

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        src = eth.src
        dst = eth.dst

        self.mac_to_port.setdefault(
            dpid,
            {}
        )

        self.mac_to_port[
            dpid
        ][
            src
        ] = in_port

        out_port = self.mac_to_port[
            dpid
        ].get(
            dst,
            ofproto.OFPP_FLOOD
        )

        telemetry = None

        if (
            self.mode
            in {
                "features",
                "inference"
            }
            and dpid
            == self.telemetry_dpid
        ):
            telemetry = self._process_telemetry(
                msg,
                pkt,
                len(
                    msg.data
                ),
                packet_in_start_ns
            )

            if (
                telemetry is not None
                and telemetry[
                    "complete"
                ]
            ):
                self._install_bypass(
                    datapath,
                    in_port,
                    eth,
                    telemetry[
                        "information"
                    ],
                    out_port
                )

        if getattr(
            msg,
            "reason",
            None
        ) == ofproto.OFPR_ACTION:
            return

        actions = self._telemetry_actions(
            datapath,
            out_port
        )

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=in_port,
                eth_src=src,
                eth_dst=dst
            )

            if (
                self.mode
                == "forwarding"
                and msg.buffer_id
                != ofproto.OFP_NO_BUFFER
            ):
                self.add_flow(
                    datapath,
                    10,
                    match,
                    actions,
                    msg.buffer_id
                )

                self.logger.info(
                    "FLOW_LEARNED "
                    "dpid=%016x src=%s dst=%s "
                    "in_port=%s out_port=%s",
                    dpid,
                    src,
                    dst,
                    in_port,
                    out_port
                )

                return

            self.add_flow(
                datapath,
                10,
                match,
                actions
            )

            self.logger.info(
                "FLOW_LEARNED "
                "dpid=%016x src=%s dst=%s "
                "in_port=%s out_port=%s",
                dpid,
                src,
                dst,
                in_port,
                out_port
            )

        data = None

        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        packet_out_actions = [
            parser.OFPActionOutput(
                out_port
            )
        ]

        packet_out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=packet_out_actions,
            data=data
        )

        datapath.send_msg(
            packet_out
        )
