# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
import logging
from types import SimpleNamespace

import unyt

from radhydropy import output
from tests.parameter_fixtures import parameter_namespace


def test_fixed_cadence_timestep_diagnostic_is_structured(caplog):
    par = parameter_namespace(
        timesim=1.0 * unyt.s,
        outdeltatime=1.0 * unyt.s,
        verbose=1,
    )
    fluid = SimpleNamespace(time_proper_code=0.0 * unyt.s)
    sim = SimpleNamespace(par=par, fluid=fluid)
    callback = output.hdf5_output_callback(
        sim,
        output_state={"outtime": 0.0 * unyt.s, "outindex": 1},
        output_writer=lambda current_sim, index: "unused.hdf5",
    )

    with caplog.at_level(logging.INFO, logger="radhydropy"):
        callback(sim, {"dt": 0.1 * unyt.s})

    record = next(record for record in caplog.records if record.msg == "hydro_step")
    assert record.event == "hydro_step"
    assert record.diagnostic_fields["runtime_time_field"] == "time_proper_code"
    assert record.diagnostic_fields["dt_runtime_code"] == 0.1 * unyt.s
