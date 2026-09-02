"""Helper utilities for the spherical stellar-wind bubble example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt
from types import SimpleNamespace

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
from radhydropy.units import CodeUnits
import weaver_analytic as wa


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


def _current_time(rout):
    simulation = getattr(rout.par, 'simulation', None)
    if simulation is not None and hasattr(simulation, 'current_time'):
        return simulation.current_time
    return rout.par.time


def _config_value(mapping, group, key, legacy):
    values = mapping.get(group)
    if isinstance(values, dict) and key in values:
        return values[key]
    return mapping[legacy]


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


class Simwrap:
    def __init__(self, icparams, code_units=None, boundary_params=None):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()
        self.par.units = SimpleNamespace(CodeUnits=code_units)
        if code_units is not None:
            self.par.unit_system = code_units.unit_system

        grid_cells = icparams['grid_cells']
        box_size = icparams['box_size'] * np.ones(1)
        self.par.mesh = SimpleNamespace(ghost_cells=0, grid_cells=grid_cells)
        self.par.simulation = SimpleNamespace(
            coordinate_system=icparams['coordinate_system'],
            current_time=icparams['current_time'] * np.ones(1),
            box_size=box_size,
        )

        self.mesh.boundary = np.linspace(
            icparams['injection_radius'],
            icparams['injection_radius'] + box_size[0],
            grid_cells + 1,
        )
        self.fluid.vel_code = icparams['velocity'] * np.ones(grid_cells)
        self.fluid.temp_code = icparams['temperature'] * np.ones(grid_cells)
        self.fluid.rho_code = icparams['initial_density'] * np.ones(grid_cells)
        self.fluid.mu = icparams['mean_molecular_weight'] * np.ones(grid_cells)

        # Match the WindSph ghost profile in a resolved active launch region.
        wind_cells = int(icparams.get('wind_injection_cells', 0))
        if (
            boundary_params is not None
            and boundary_params.get('condition') == 'WindSph'
            and wind_cells > 0
        ):
            centers = 0.5 * (
                self.mesh.boundary[:-1] + self.mesh.boundary[1:]
            )
            launch = np.arange(grid_cells) < wind_cells
            radius = centers[launch]
            reference_radius = icparams['injection_radius']
            wind_density = boundary_params['outflow_density'] * (
                reference_radius / radius
            ) ** 2
            self.fluid.rho_code[launch] = wind_density
            self.fluid.vel_code[launch] = boundary_params['outflow_velocity']
            self.fluid.temp_code[launch] = boundary_params['outflow_temperature']
            self.fluid.mu[launch] = boundary_params['outflow_mu']


def load_snapshot(outfilename, icparams, runparams):
    """Load an output snapshot into a lightweight simulation wrapper."""

    rout = Simwrap(icparams)
    code_units_obj = CodeUnits.from_mapping(runparams['units']['CodeUnits'])
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    # Solver outputs retain ghost zones, whereas the example diagnostics are
    # defined on the physical domain.  Trim every cell-centered fluid field
    # and the corresponding faces before calculating profiles or shell
    # diagnostics; otherwise ghost states can be mistaken for the swept-up
    # shell and produce discontinuous pressure histories.
    first = int(runparams.get('mesh', {}).get('ghost_cells', 0))
    configured_count = int(runparams.get('mesh', {}).get(
        'grid_cells', icparams['grid_cells']
    ))
    boundary_count = len(rout.mesh.boundary) - 1
    # Permit reduced-resolution diagnostic runs whose output count is lower
    # than the production value still present in the YAML configuration.
    count = min(configured_count, boundary_count - 2 * first)
    if boundary_count >= first + count and boundary_count != count:
        stop = first + count
        rout.mesh.boundary = rout.mesh.boundary[first:stop + 1]
        for name, value in vars(rout.fluid).items():
            try:
                value_length = len(value)
            except TypeError:
                continue
            if value_length == boundary_count:
                setattr(rout.fluid, name, value[first:stop])
    rout.par.simulation.current_time = unyt.unyt_array(np.asarray(rout.par.simulation.current_time, dtype=float), code_units_obj.time_unit)
    rout.par.simulation.box_size = unyt.unyt_array(np.asarray(rout.par.simulation.box_size, dtype=float), code_units_obj.length_unit)
    rout.mesh.boundary = unyt.unyt_array(np.asarray(rout.mesh.boundary, dtype=float), code_units_obj.length_unit)
    rout.fluid.vel_code = unyt.unyt_array(np.asarray(rout.fluid.vel_code, dtype=float), code_units_obj.velocity_unit)
    rout.fluid.temp_code = unyt.unyt_array(np.asarray(rout.fluid.temp_code, dtype=float), code_units_obj.temperature_unit)
    rout.fluid.rho_code = unyt.unyt_array(np.asarray(rout.fluid.rho_code, dtype=float), code_units_obj.density_unit)
    rout.fluid.mu = np.asarray(rout.fluid.mu, dtype=float)
    if hasattr(rout.fluid, 'xHI'):
        rout.fluid.xHI = np.asarray(rout.fluid.xHI, dtype=float)
    if hasattr(rout.fluid, 'ngamma_code'):
        rout.fluid.ngamma_code = unyt.unyt_array(
            np.asarray(rout.fluid.ngamma_code, dtype=float),
            code_units_obj.number_density_unit,
        )
    return rout


def numerical_forward_shock_radius(rout, search_fraction=0.1):
    """Estimate the forward-shock radius from the steepest pressure drop."""

    coordinate = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    pressure = (
        rout.fluid.rho_code
        / (rout.fluid.mu * unyt.mp)
        * unyt.kb
        * rout.fluid.temp_code
    ).to(unyt.dyn / unyt.cm**2)

    coordinate_values = coordinate.to_value(coordinate.units)
    pressure_values = pressure.to_value(pressure.units)

    if pressure_values.size < 3:
        return None
    if np.ptp(pressure_values) == 0.0:
        return None

    mask = coordinate_values >= 0.0
    coordinate_values = coordinate_values[mask]
    coordinate = coordinate[mask]
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
    return coordinate[shock_index]


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

    coordinate = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    density = rout.fluid.rho_code

    coordinate_values = coordinate.to_value(coordinate.units)
    density_values = density.to_value(density.units)
    mask = coordinate_values >= 0.0
    coordinate_values = coordinate_values[mask]
    density_values = density_values[mask]
    coordinate = coordinate[mask]

    if minimum_radius is not None:
        minimum_radius_value = minimum_radius.to_value(coordinate.units)
        keep = coordinate_values >= minimum_radius_value
        coordinate_values = coordinate_values[keep]
        density_values = density_values[keep]
        coordinate = coordinate[keep]

    if density_values.size < 2:
        return None

    threshold = (
        ambient_density.to_value(density.units)
        * float(np.asarray(threshold_factor).reshape(-1)[0])
    )
    # Equality is the ambient state itself, not shell compression.  Using
    # ``>=`` makes an unperturbed ambient profile look like a shell beginning
    # at the first active cell when the threshold factor is 1.
    above = density_values > threshold
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
        return coordinate[0]

    x0 = coordinate_values[edge_index - 1]
    x1 = coordinate_values[edge_index]
    y0 = density_values[edge_index - 1]
    y1 = density_values[edge_index]
    if y1 == y0:
        return coordinate[edge_index]

    fraction = (threshold - y0) / (y1 - y0)
    fraction = np.clip(fraction, 0.0, 1.0)
    radius = x0 + fraction * (x1 - x0)
    return radius * coordinate.units


def shell_search_minimum_radius(rout, icparams):
    """Return the outer edge of a resolved wind launch region."""

    wind_cells = int(icparams.get('wind_injection_cells', 0))
    if wind_cells <= 0:
        return None
    first = int(getattr(rout.par.mesh, 'ghost_cells', 0))
    dx = abs(rout.mesh.boundary[first + 1] - rout.mesh.boundary[first])
    return icparams['injection_radius'] + wind_cells * dx


def weaver_forward_shock_radius(rout, icparams, runparams):
    """Return the Weaver shock radius for a loaded snapshot."""

    return wa.shock_radius(
        _current_time(rout),
        icparams.get('initial_density', icparams.get('rhoini')),
        _config_value(runparams, 'boundary', 'outflow_density', 'rho_outflow'),
        _config_value(runparams, 'boundary', 'outflow_velocity', 'vel_outflow'),
        icparams.get('injection_radius', icparams.get('rinj')),
    )


def _snapshot_coordinate(rout, xunit=unyt.pc):
    """Return nonnegative cell-center coordinates for a snapshot."""

    coordinate = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    coordinate_values = coordinate.to_value(xunit)
    nonnegative = coordinate_values >= 0.0
    return coordinate[nonnegative], coordinate_values[nonnegative]


def plot_density_snapshot(ax, rout, **kwargs):
    """Plot one density snapshot on a supplied axis."""

    plt.sca(ax)
    rplot1d(rout, yquan='rho_code', showhalf=0, showfig=0, **kwargs)
    ax.set_yscale('log')


def plot_temperature_snapshot(ax, rout, **kwargs):
    """Plot one temperature snapshot on a supplied axis."""

    plt.sca(ax)
    rplot1d(rout, yquan='temp_code', showhalf=0, showfig=0, **kwargs)
    ax.set_yscale('log')


def plot_profile_snapshot(ax, rout, yquan, xunit=unyt.pc, **kwargs):
    """Plot one radial profile on a supplied axis using ``xunit``."""

    coordinate = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    coordinate_values = coordinate.to_value(xunit)
    nonnegative = coordinate_values >= 0.0
    ax.plot(
        coordinate_values[nonnegative],
        getattr(rout.fluid, yquan).to_value(getattr(rout.fluid, yquan).units)[
            nonnegative
        ],
        **kwargs,
    )
    ax.set_yscale('log')


def make_profile_figure(snapshots, icparams, runparams):
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
            'rho_code',
            ls='none',
            marker='o',
            mfc='none',
            markevery=5,
            color=color,
        )
        plot_profile_snapshot(
            ax_temperature,
            rout,
            'temp_code',
            ls='none',
            marker='o',
            mfc='none',
            markevery=5,
            color=color,
        )
        if _current_time(rout) > 0 * _current_time(rout).units:
            shock_radius = weaver_forward_shock_radius(rout, icparams, runparams)
            shock_value = shock_radius.to_value(icparams['injection_radius'].units).item()
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


def make_radius_figure(snapshots, icparams, runparams):
    """Build the cavity-side inner-shell-edge radius evolution figure."""

    figure, ax_radius = plt.subplots(1, 1, figsize=(8.5, 6.0))
    numerical_times = []
    numerical_radii = []
    weaver_times = []
    weaver_radii = []
    shell_threshold_factor = runparams.get('example', {}).get('shell_edge_density_threshold_factor', runparams.get('shell_edge_density_threshold_factor', 1.0))

    for rout in snapshots:
        if _current_time(rout) <= 0 * _current_time(rout).units:
            continue
        numerical_radius = shell_inner_edge_radius(
            rout,
            icparams.get('initial_density', icparams.get('rhoini')),
            shell_threshold_factor,
            minimum_radius=shell_search_minimum_radius(rout, icparams),
        )
        if numerical_radius is None:
            continue
        weaver_radius = weaver_forward_shock_radius(rout, icparams, runparams)
        time_myr = _current_time(rout).to_value(unyt.Myr)
        numerical_times.append(time_myr)
        numerical_radii.append(numerical_radius.to_value(unyt.pc))
        weaver_times.append(time_myr)
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


def numerical_bubble_pressure(rout, shell_radius):
    """Estimate the bubble pressure from a cavity-side annulus."""

    coordinate = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    coordinate_values = coordinate.to_value(unyt.pc)
    nonnegative = coordinate_values >= 0.0
    coordinate_values = coordinate_values[nonnegative]
    shell_radius_value = shell_radius.to_value(unyt.pc)
    pressure = (
        rout.fluid.rho_code
        / (rout.fluid.mu * unyt.mp)
        * unyt.kb
        * rout.fluid.temp_code
    ).to(unyt.dyn / unyt.cm**2)
    pressure_values = pressure.to_value(pressure.units)[nonnegative]

    if pressure_values.size < 2:
        return None

    shell_width = max(0.05 * shell_radius_value, 0.1)
    cavity_band = (coordinate_values < shell_radius_value) & (
        coordinate_values >= shell_radius_value - shell_width
    )
    if not np.any(cavity_band):
        return None

    return unyt.unyt_quantity(np.median(pressure_values[cavity_band]), pressure.units)


def collect_shell_diagnostics(snapshots, icparams, runparams):
    """Collect shell radius, velocity, and pressure comparison data."""

    shell_threshold_factor = runparams.get('example', {}).get('shell_edge_density_threshold_factor', runparams.get('shell_edge_density_threshold_factor', 1.0))
    times = []
    radii = []
    pressures = []

    for rout in snapshots:
        if _current_time(rout) <= 0 * _current_time(rout).units:
            continue
        shell_radius = shell_inner_edge_radius(
            rout,
            icparams.get('initial_density', icparams.get('rhoini')),
            shell_threshold_factor,
            minimum_radius=shell_search_minimum_radius(rout, icparams),
        )
        if shell_radius is None:
            continue
        bubble_pressure = numerical_bubble_pressure(rout, shell_radius)
        if bubble_pressure is None:
            continue
        times.append(_current_time(rout))
        radii.append(shell_radius)
        pressures.append(bubble_pressure)

    if not times:
        return None

    times = unyt.unyt_array([time.to_value(unyt.Myr) for time in times], unyt.Myr)
    radii = unyt.unyt_array([radius.to_value(unyt.pc) for radius in radii], unyt.pc)
    pressures = unyt.unyt_array(
        [pressure.to_value(unyt.dyn / unyt.cm**2) for pressure in pressures],
        unyt.dyn / unyt.cm**2,
    )
    time_values = np.array([float(time.to_value(unyt.Myr)) for time in times], dtype=float)
    radius_values = np.array(
        [float(radius.to_value(unyt.pc)) for radius in radii],
        dtype=float,
    )
    velocities = np.gradient(radius_values, time_values)
    velocities = unyt.unyt_array(velocities, unyt.pc / unyt.Myr).to(unyt.km / unyt.s)
    weaver_radii = []
    weaver_velocities = []
    weaver_pressures = []
    for time in times:
        radius, velocity, pressure = wa.weaver_solution(
            time,
            icparams.get('initial_density', icparams.get('rhoini')),
            _config_value(runparams, 'boundary', 'outflow_density', 'rho_outflow'),
            _config_value(runparams, 'boundary', 'outflow_velocity', 'vel_outflow'),
            icparams.get('injection_radius', icparams.get('rinj')),
        )
        weaver_radii.append(radius.to_value(unyt.pc))
        weaver_velocities.append(velocity.to_value(unyt.km / unyt.s))
        weaver_pressures.append(pressure.to_value(unyt.dyn / unyt.cm**2))

    return {
        'times': times.to(unyt.Myr),
        'radii': radii.to(unyt.pc),
        'velocities': velocities,
        'pressures': pressures,
        'weaver_radii': unyt.unyt_array(weaver_radii, unyt.pc),
        'weaver_velocities': unyt.unyt_array(weaver_velocities, unyt.km / unyt.s),
        'weaver_pressures': unyt.unyt_array(weaver_pressures, unyt.dyn / unyt.cm**2),
    }


def make_velocity_figure(snapshots, icparams, runparams):
    """Build the shock-velocity comparison figure."""

    diagnostics = collect_shell_diagnostics(snapshots, icparams, runparams)
    if diagnostics is None:
        return None

    figure, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))
    ax.plot(
        diagnostics['times'].to_value(unyt.Myr),
        diagnostics['velocities'].to_value(unyt.km / unyt.s),
        color='k',
        lw=2.0,
        marker='o',
        label='simulation',
    )
    ax.plot(
        diagnostics['times'].to_value(unyt.Myr),
        diagnostics['weaver_velocities'].to_value(unyt.km / unyt.s),
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


def make_pressure_figure(snapshots, icparams, runparams):
    """Build the bubble-pressure comparison figure."""

    diagnostics = collect_shell_diagnostics(snapshots, icparams, runparams)
    if diagnostics is None:
        return None

    figure, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))
    ax.plot(
        diagnostics['times'].to_value(unyt.Myr),
        diagnostics['pressures'].to_value(unyt.dyn / unyt.cm**2),
        color='k',
        lw=2.0,
        marker='o',
        label='simulation',
    )
    ax.plot(
        diagnostics['times'].to_value(unyt.Myr),
        diagnostics['weaver_pressures'].to_value(unyt.dyn / unyt.cm**2),
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


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = load_snapshot(outfilename, icparams, runparams)
    plot_density_snapshot(plt.gca(), rout, **kwargs)
    if np.all(_current_time(rout) > 0 * _current_time(rout).units):
        shock_radius = weaver_forward_shock_radius(rout, icparams, runparams)
        plt.axvline(
            x=shock_radius.to_value(icparams['injection_radius'].units).item(),
            color=kwargs['color'],
            ls='dashed',
        )
