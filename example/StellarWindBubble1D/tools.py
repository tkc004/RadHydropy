"""Helper utilities for the spherical stellar-wind bubble example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
import radhydropy.io as rio
from radhydropy.initial_condition_writer import InitialConditionWriter
from radhydropy.units import CodeUnits
import weaver_analytic as wa


def _time_proper(rout):
    return unyt.unyt_quantity(
        float(np.asarray(rout.fluid.time_proper_code)),
        rout.par.units.CodeUnits.time_unit,
    )


def set_plot_style():
    plt.rcParams.update(
        {
            'axes.labelsize': 24,
            'axes.titlesize': 24,
            'font.size': 24,
            'legend.fontsize': 18,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'xtick.top': True,
            'ytick.right': True,
            'xtick.bottom': True,
            'ytick.left': True,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'figure.figsize': (12.0, 6.0),
            'lines.markersize': 5,
            'lines.linewidth': 2.5,
        }
    )


def build_initial_condition(config):
    """Build the proper-coordinate IC through the shared writer boundary."""
    initial_config = config['initial_condition']
    code_units = config.get('_code_units') or CodeUnits.from_mapping(
        config['par']['units']['CodeUnits']
    )
    grid_cells = int(initial_config['grid_cells'])
    boundary_proper_unyt = np.linspace(
        initial_config['radius_injection_proper'],
        initial_config['radius_injection_proper'] + initial_config['box_size_proper'],
        grid_cells + 1,
    )
    writer = InitialConditionWriter(
        par_config=config['par'],
        code_units=code_units,
        ic_config=initial_config,
    )
    writer.box_size = writer.radquantity(
        initial_config['radius_injection_proper']
        + initial_config['box_size_proper']
    )
    writer.mesh.boundary_radarray = writer.radarray(boundary_proper_unyt)
    rho_proper_unyt = initial_config['rho_proper'] * np.ones(grid_cells)
    vel_proper_unyt = initial_config['vel_proper'] * np.ones(grid_cells)
    temp_proper_unyt = initial_config['temperature_proper'] * np.ones(grid_cells)
    writer.fluid.rho_radarray = writer.radarray(rho_proper_unyt)
    writer.fluid.vel_radarray = writer.radarray(vel_proper_unyt)
    writer.fluid.temp_radarray = writer.radarray(temp_proper_unyt)
    writer.fluid.mu = initial_config['mean_molecular_weight'] * np.ones(grid_cells)
    return writer

def load_output_state(outfilename, config):
    """Load an output snapshot through the typed nested-config boundary."""
    initial_config = config['initial_condition']
    rout = rio.loadhdf5(config, outfilename)
    # Solver outputs retain ghost zones, whereas the example diagnostics are
    # defined on the physical domain.  Trim every cell-centered fluid field
    # and the corresponding faces before calculating profiles or shell
    # diagnostics; otherwise ghost states can be mistaken for the swept-up
    # shell and produce discontinuous pressure histories.
    first = int(config["par"].get('mesh', {}).get('ghost_cells', 0))
    configured_count = int(config["par"].get('mesh', {}).get(
        'grid_cells', initial_config['grid_cells']
    ))
    boundary_count = len(rout.mesh.boundary_radarray) - 1
    # Output snapshots contain ghost cells; the initial-condition snapshot
    # contains active cells only and is therefore already diagnostic-ready.
    count = configured_count
    if boundary_count == configured_count + 2 * first:
        stop = first + count
        rout.mesh.boundary_radarray = rout.mesh.boundary_radarray[first:stop + 1]
        for field_name in ('rho_radarray', 'vel_radarray', 'temp_radarray'):
            field = getattr(rout.fluid, field_name)
            setattr(rout.fluid, field_name, field[first:stop])
        rout.fluid.mu = np.asarray(rout.fluid.mu)[first:stop]
    return rout


def _boundary_proper_unyt(rout):
    """Return the restored proper-coordinate boundary for diagnostics."""
    if hasattr(rout.mesh, 'boundary_radarray'):
        return rout.mesh.boundary_radarray
    return rout.mesh.boundary_proper_code


def _rho_proper_unyt(rout):
    """Return the restored proper density for diagnostics."""
    if hasattr(rout.fluid, 'rho_radarray'):
        return rout.fluid.rho_radarray
    return rout.fluid.rho_proper_code


def _temp_proper_unyt(rout):
    """Return the restored proper temperature for diagnostics."""
    if hasattr(rout.fluid, 'temp_radarray'):
        return rout.fluid.temp_radarray
    return rout.fluid.temp_proper_code


def numerical_forward_shock_radius(rout, search_fraction=0.1):
    """Estimate the forward-shock radius from the steepest pressure drop."""

    boundary_proper_unyt = _boundary_proper_unyt(rout)
    x_proper_code = 0.5 * (boundary_proper_unyt[1:] + boundary_proper_unyt[:-1])
    rho_proper_cgs_g_cm3 = _rho_proper_unyt(rout).to(unyt.g / unyt.cm**3)
    temperature_cgs_K = _temp_proper_unyt(rout).to(unyt.K)
    pressure_bubble_proper_unyt = (
        rho_proper_cgs_g_cm3
        / (rout.fluid.mu * unyt.mp)
        * unyt.kb
        * temperature_cgs_K
    ).to(unyt.dyn / unyt.cm**2)

    coordinate_values = x_proper_code.to_value(x_proper_code.units)
    pressure_values = pressure_bubble_proper_unyt.to_value(pressure_bubble_proper_unyt.units)

    if pressure_values.size < 3:
        return None
    if np.ptp(pressure_values) == 0.0:
        return None

    mask = coordinate_values >= 0.0
    coordinate_values = coordinate_values[mask]
    x_proper_code = x_proper_code[mask]
    pressure_values = pressure_values[mask]
    if pressure_values.size < 3:
        return None

    search_start = max(5, int(search_fraction * pressure_values.size))
    search_start = min(search_start, pressure_values.size - 2)

    gradient = np.gradient(pressure_values, coordinate_values)
    shock_slice = gradient[search_start:]
    if shock_slice.size == 0:
        return None
    shock_index = search_start + int(np.argmin(shock_slice))
    return x_proper_code[shock_index]


def format_density_threshold_factor(threshold_factor):
    """Return a compact label for the shell-edge density factor."""

    factor_value = float(np.asarray(threshold_factor).reshape(-1)[0])
    return f"{factor_value:.2g}"


def shell_inner_edge_radius(
    rout,
    ambient_density,
    threshold_factor=1.0,
    minimum_radius=None,
):
    """Estimate the cavity-side edge of the outer swept-up shell.

    The resolved wind launch region and contact discontinuity can create
    several separate density excursions above the ambient threshold.  The
    forward swept-up shell is the outermost such excursion, not necessarily
    the first one encountered after the launch region.
    """

    boundary_proper_unyt = _boundary_proper_unyt(rout)
    x_proper_code = 0.5 * (boundary_proper_unyt[1:] + boundary_proper_unyt[:-1])
    rho_proper_unyt = _rho_proper_unyt(rout)

    coordinate_values = x_proper_code.to_value(x_proper_code.units)
    rho_proper_values = rho_proper_unyt.to_value(rho_proper_unyt.units)
    mask = coordinate_values >= 0.0
    coordinate_values = coordinate_values[mask]
    rho_proper_values = rho_proper_values[mask]
    x_proper_code = x_proper_code[mask]

    if minimum_radius is not None:
        minimum_radius_value = minimum_radius.to_value(x_proper_code.units)
        keep = coordinate_values >= minimum_radius_value
        coordinate_values = coordinate_values[keep]
        rho_proper_values = rho_proper_values[keep]
        x_proper_code = x_proper_code[keep]

    if rho_proper_values.size < 2:
        return None

    threshold = (
        ambient_density.to_value(rho_proper_unyt.units)
        * float(np.asarray(threshold_factor).reshape(-1)[0])
    )
    # Equality is the ambient state itself, not shell compression.  Using
    # ``>=`` makes an unperturbed ambient profile look like a shell beginning
    # at the first active cell when the threshold factor is 1.
    above = rho_proper_values > threshold
    if not np.any(above):
        return None

    starts = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    if starts.size == 0:
        return None

    if above[0]:
        # The inner wind can itself be above the ambient threshold.  Skip
        # that initial region and use the first subsequent crossing, which is
        # the cavity-side edge of the swept-up shell.
        below = np.flatnonzero(~above)
        if below.size == 0:
            return None
        search_start = int(below[0] + 1)
        candidate_starts = starts[starts >= search_start]
        if candidate_starts.size == 0:
            return None
        edge_index = int(candidate_starts[0])
    else:
        edge_index = int(starts[0])

    if edge_index == 0:
        return x_proper_code[0]

    x0 = coordinate_values[edge_index - 1]
    x1 = coordinate_values[edge_index]
    y0 = rho_proper_values[edge_index - 1]
    y1 = rho_proper_values[edge_index]
    if y1 == y0:
        return x_proper_code[edge_index]

    fraction = (threshold - y0) / (y1 - y0)
    fraction = np.clip(fraction, 0.0, 1.0)
    shell_radius_proper_code = x0 + fraction * (x1 - x0)
    return shell_radius_proper_code * x_proper_code.units


def weaver_forward_shock_radius(rout, config):
    """Return the Weaver shock radius for a loaded snapshot."""

    return wa.shock_radius(
        _time_proper(rout),
        config['initial_condition']['rho_proper'],
        config['par']['boundary']['rho_outflow_proper'],
        config['par']['boundary']['vel_outflow_proper'],
        config['initial_condition']['radius_injection_proper'],
    )


def _snapshot_coordinate(rout, xunit=unyt.pc):
    """Return nonnegative cell-center coordinates for a snapshot."""

    boundary_proper_unyt = _boundary_proper_unyt(rout)
    x_proper_code = 0.5 * (boundary_proper_unyt[1:] + boundary_proper_unyt[:-1])
    coordinate_values = x_proper_code.to_value(xunit)
    nonnegative = coordinate_values >= 0.0
    return x_proper_code[nonnegative], coordinate_values[nonnegative]


def plot_density_snapshot(ax, rout, **kwargs):
    """Plot one density snapshot on a supplied axis."""

    plt.sca(ax)
    x_proper_code, coordinate_values = _snapshot_coordinate(rout)
    rho_proper_unyt = _rho_proper_unyt(rout).to(unyt.g / unyt.cm**3)
    ax.plot(coordinate_values, rho_proper_unyt.to_value(unyt.g / unyt.cm**3), **kwargs)
    ax.set_yscale('log')


def plot_temperature_snapshot(ax, rout, **kwargs):
    """Plot one temperature snapshot on a supplied axis."""

    plt.sca(ax)
    x_proper_code, coordinate_values = _snapshot_coordinate(rout)
    temperature_proper_unyt = _temp_proper_unyt(rout).to(unyt.K)
    ax.plot(coordinate_values, temperature_proper_unyt.to_value(unyt.K), **kwargs)
    ax.set_yscale('log')


def plot_profile_snapshot(ax, rout, yquan, xunit=unyt.pc, **kwargs):
    """Plot one radial profile on a supplied axis using ``xunit``."""

    boundary_proper_unyt = _boundary_proper_unyt(rout)
    x_proper_code = 0.5 * (boundary_proper_unyt[1:] + boundary_proper_unyt[:-1])
    coordinate_values = x_proper_code.to_value(xunit)
    nonnegative = coordinate_values >= 0.0
    ax.plot(
        coordinate_values[nonnegative],
        getattr(rout.fluid, yquan).to_value(getattr(rout.fluid, yquan).units)[
            nonnegative
        ],
        **kwargs,
    )
    ax.set_yscale('log')


def make_profile_figure(snapshots, config):
    """Build the stacked density/temperature comparison figure."""

    figure, (ax_density, ax_temperature) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10.0, 11.0),
    )
    ax_density.set_yscale('log')
    ax_temperature.set_yscale('log')

    for index, rout in enumerate(snapshots):
        color = next(ax_density._get_lines.prop_cycler)['color']
        plot_profile_snapshot(
            ax_density,
            rout,
            'rho_radarray',
            ls='none',
            marker='o',
            mfc='none',
            markevery=5,
            color=color,
        )
        plot_profile_snapshot(
            ax_temperature,
            rout,
            'temp_radarray',
            ls='none',
            marker='o',
            mfc='none',
            markevery=5,
            color=color,
        )
        if _time_proper(rout) > 0 * _time_proper(rout).units:
            shock_radius = weaver_forward_shock_radius(rout, config)
            shock_value = shock_radius.to_value(config['initial_condition']['radius_injection_proper'].units).item()
            for ax in (ax_density, ax_temperature):
                ax.axvline(
                    x=shock_value,
                    color=color,
                    ls='--',
                    alpha=0.35,
                )

    ax_density.set_title('Density profile')
    ax_density.set_xlabel(r'$r$ [pc]')
    ax_density.set_ylabel(r'$\rho$ [g cm$^{-3}$]')
    ax_temperature.set_title('Temperature profile')
    ax_temperature.set_xlabel(r'$r$ [pc]')
    ax_temperature.set_ylabel(r'$T$ [K]')
    figure.tight_layout()
    return figure


def make_radius_figure(snapshots, config):
    """Build the cavity-side inner-shell-edge radius evolution figure."""

    figure, ax_radius = plt.subplots(1, 1, figsize=(8.5, 6.0))
    numerical_times = []
    numerical_radii = []
    weaver_times = []
    weaver_radii = []
    initial_config = config['initial_condition']

    shell_threshold_factor = config['example'].get('shell_edge_density_threshold_factor', 1.0)

    for rout in snapshots:
        if _time_proper(rout) <= 0 * _time_proper(rout).units:
            continue
        numerical_radius = shell_inner_edge_radius(
            rout,
            initial_config['rho_proper'],
            shell_threshold_factor,
        )
        if numerical_radius is None:
            continue
        weaver_radius = weaver_forward_shock_radius(rout, config)
        time_proper_Myr = _time_proper(rout).to_value(unyt.Myr)
        numerical_times.append(time_proper_Myr)
        numerical_radii.append(numerical_radius.to_value(unyt.pc))
        weaver_times.append(time_proper_Myr)
        weaver_radii.append(weaver_radius.to_value(unyt.pc))

    ax_radius.plot(
        numerical_times,
        numerical_radii,
        color='k',
        lw=2.0,
        marker='o',
        label=(
            'cavity-side inner edge '
            f'(rho > {format_density_threshold_factor(shell_threshold_factor)} '
            r'$\rho_{\rm amb}$)'
        ),
    )
    ax_radius.plot(
        weaver_times,
        weaver_radii,
        color='k',
        lw=2.0,
        ls='--',
        marker='x',
        label='Weaver 1977',
    )
    ax_radius.set_title('Cavity-side inner-shell-edge radius evolution')
    ax_radius.set_xlabel('Time [Myr]')
    ax_radius.set_ylabel(r'$R_{\rm in}$ [pc]')
    ax_radius.legend(loc='best')
    ax_radius.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def numerical_bubble_pressure(rout, radius_shell_proper_unyt):
    """Estimate the bubble pressure from a cavity-side annulus."""

    boundary_proper_unyt = _boundary_proper_unyt(rout)
    x_proper_code = 0.5 * (boundary_proper_unyt[1:] + boundary_proper_unyt[:-1])
    coordinate_values = x_proper_code.to_value(unyt.pc)
    nonnegative = coordinate_values >= 0.0
    coordinate_values = coordinate_values[nonnegative]
    radius_shell_proper_pc = radius_shell_proper_unyt.to_value(unyt.pc)
    rho_proper_cgs_g_cm3 = _rho_proper_unyt(rout).to(unyt.g / unyt.cm**3)
    temperature_cgs_K = _temp_proper_unyt(rout).to(unyt.K)
    pressure_bubble_proper_unyt = (
        rho_proper_cgs_g_cm3
        / (rout.fluid.mu * unyt.mp)
        * unyt.kb
        * temperature_cgs_K
    ).to(unyt.dyn / unyt.cm**2)
    pressure_values = pressure_bubble_proper_unyt.to_value(pressure_bubble_proper_unyt.units)[nonnegative]

    if pressure_values.size < 2:
        return None

    shell_width = max(0.05 * radius_shell_proper_pc, 0.1)
    cavity_band = (coordinate_values < radius_shell_proper_pc) & (
        coordinate_values >= radius_shell_proper_pc - shell_width
    )
    if not np.any(cavity_band):
        return None

    return unyt.unyt_quantity(np.median(pressure_values[cavity_band]), pressure_bubble_proper_unyt.units)


def collect_shell_diagnostics(snapshots, config):
    """Collect shell radius, velocity, and pressure comparison data."""

    initial_config = config['initial_condition']
    shell_threshold_factor = config['example'].get('shell_edge_density_threshold_factor', 1.0)
    times_proper_unyt = []
    radii_shell_proper_unyt = []
    pressures_bubble_proper_unyt = []

    for rout in snapshots:
        if _time_proper(rout) <= 0 * _time_proper(rout).units:
            continue
        radius_shell_proper_unyt = shell_inner_edge_radius(
            rout,
            initial_config['rho_proper'],
            shell_threshold_factor,
        )
        if radius_shell_proper_unyt is None:
            continue
        bubble_pressure = numerical_bubble_pressure(rout, radius_shell_proper_unyt)
        if bubble_pressure is None:
            continue
        times_proper_unyt.append(_time_proper(rout))
        radii_shell_proper_unyt.append(radius_shell_proper_unyt)
        pressures_bubble_proper_unyt.append(bubble_pressure)

    if not times_proper_unyt:
        return None

    times_proper_unyt = unyt.unyt_array([time_proper_unyt.to_value(unyt.Myr) for time_proper_unyt in times_proper_unyt], unyt.Myr)
    radii_shell_proper_unyt = unyt.unyt_array([radius_shell_proper_unyt.to_value(unyt.pc) for radius_shell_proper_unyt in radii_shell_proper_unyt], unyt.pc)
    pressures_bubble_proper_unyt = unyt.unyt_array(
        [pressure_bubble_proper_unyt.to_value(unyt.dyn / unyt.cm**2) for pressure_bubble_proper_unyt in pressures_bubble_proper_unyt],
        unyt.dyn / unyt.cm**2,
    )
    time_proper_Myr = np.array([float(time_proper_unyt.to_value(unyt.Myr)) for time_proper_unyt in times_proper_unyt], dtype=float)
    radius_shell_proper_pc = np.array(
        [float(radius_shell_proper_unyt.to_value(unyt.pc)) for radius_shell_proper_unyt in radii_shell_proper_unyt],
        dtype=float,
    )
    vel_shell_proper_km_s = np.gradient(radius_shell_proper_pc, time_proper_Myr)
    vel_shell_proper_km_s = unyt.unyt_array(vel_shell_proper_km_s, unyt.pc / unyt.Myr).to(unyt.km / unyt.s)
    weaver_radii = []
    weaver_velocities = []
    weaver_pressures = []
    for time_proper_unyt in times_proper_unyt:
        radius_shock_proper_unyt, vel_shock_proper_unyt, pressure_bubble_proper_unyt = wa.weaver_solution(
            time_proper_unyt,
            initial_config['rho_proper'],
            config['par']['boundary']['rho_outflow_proper'],
            config['par']['boundary']['vel_outflow_proper'],
            initial_config['radius_injection_proper'],
        )
        weaver_radii.append(radius_shock_proper_unyt.to_value(unyt.pc))
        weaver_velocities.append(vel_shock_proper_unyt.to_value(unyt.km / unyt.s))
        weaver_pressures.append(pressure_bubble_proper_unyt.to_value(unyt.dyn / unyt.cm**2))

    return {
        'times_proper_Myr': times_proper_unyt.to(unyt.Myr),
        'radii_shell_proper_pc': radii_shell_proper_unyt.to(unyt.pc),
        'vel_shell_proper_km_s': vel_shell_proper_km_s,
        'pressures_bubble_proper_cgs_dyn_cm2': pressures_bubble_proper_unyt,
        'weaver_radius_shock_proper_pc': unyt.unyt_array(weaver_radii, unyt.pc),
        'weaver_vel_shock_proper_km_s': unyt.unyt_array(weaver_velocities, unyt.km / unyt.s),
        'weaver_pressure_bubble_proper_cgs_dyn_cm2': unyt.unyt_array(weaver_pressures, unyt.dyn / unyt.cm**2),
    }


def make_velocity_figure(snapshots, config):
    """Build the shock-velocity comparison figure."""

    diagnostics = collect_shell_diagnostics(snapshots, config)
    if diagnostics is None:
        return None

    figure, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))
    ax.plot(
        diagnostics['times_proper_Myr'].to_value(unyt.Myr),
        diagnostics['vel_shell_proper_km_s'].to_value(unyt.km / unyt.s),
        color='k',
        lw=2.0,
        marker='o',
        label='simulation',
    )
    ax.plot(
        diagnostics['times_proper_Myr'].to_value(unyt.Myr),
        diagnostics['weaver_vel_shock_proper_km_s'].to_value(unyt.km / unyt.s),
        color='k',
        lw=2.0,
        ls='--',
        marker='x',
        label='Weaver 1977',
    )
    ax.set_title('Shock velocity evolution')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$V_{\rm shock}$ [km s$^{-1}$]')
    ax.legend(loc='best')
    ax.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def make_pressure_figure(snapshots, config):
    """Build the bubble-pressure comparison figure."""

    diagnostics = collect_shell_diagnostics(snapshots, config)
    if diagnostics is None:
        return None

    figure, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))
    ax.plot(
        diagnostics['times_proper_Myr'].to_value(unyt.Myr),
        diagnostics['pressures_bubble_proper_cgs_dyn_cm2'].to_value(unyt.dyn / unyt.cm**2),
        color='k',
        lw=2.0,
        marker='o',
        label='simulation',
    )
    ax.plot(
        diagnostics['times_proper_Myr'].to_value(unyt.Myr),
        diagnostics['weaver_pressure_bubble_proper_cgs_dyn_cm2'].to_value(unyt.dyn / unyt.cm**2),
        color='k',
        lw=2.0,
        ls='--',
        marker='x',
        label='Weaver 1977',
    )
    ax.set_yscale('log')
    ax.set_title('Bubble pressure evolution')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel(r'$P_{\rm bubble}$ [dyn cm$^{-2}$]')
    ax.legend(loc='best')
    ax.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_snapshot(outfilename, config, **kwargs):
    rout = load_output_state(outfilename, config)
    plot_density_snapshot(plt.gca(), rout, **kwargs)
    if np.all(_time_proper(rout) > 0 * _time_proper(rout).units):
        shock_radius = weaver_forward_shock_radius(rout, config)
        plt.axvline(
        x=shock_radius.to_value(config['initial_condition']['radius_injection_proper'].units).item(),
            color=kwargs['color'],
            ls='dashed',
        )
