"""Helper utilities for the spherical stellar-wind bubble example."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import unyt

from radhydropy.analysis import rplot1d
import radhydropy.io as rio
import weaver_analytic as wa


class Par:
    pass


class Mesh:
    pass


class Fluid:
    pass


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
    def __init__(self, icparams):
        self.par = Par()
        self.mesh = Mesh()
        self.fluid = Fluid()

        self.par.nogrid = icparams['nogrid']
        self.par.coordsys = icparams['coordsys']
        self.par.boxsize = icparams['boxsize'] * np.ones(1)
        self.par.time = icparams['time'] * np.ones(1)

        self.mesh.boundary = np.linspace(
            icparams['rinj'],
            icparams['rinj'] + self.par.boxsize[0],
            self.par.nogrid + 1,
        )
        self.fluid.vel = icparams['vini'] * np.ones(self.par.nogrid)
        self.fluid.temp = icparams['tempini'] * np.ones(self.par.nogrid)
        self.fluid.rho = icparams['rhoini'] * np.ones(self.par.nogrid)
        self.fluid.mu = icparams['muini'] * np.ones(self.par.nogrid)


def load_snapshot(outfilename, icparams):
    """Load an output snapshot into a lightweight simulation wrapper."""

    rout = Simwrap(icparams)
    rio.readhdf5(rout.par, rout.mesh, rout.fluid, outfilename)
    return rout


def numerical_forward_shock_radius(rout, search_fraction=0.1):
    """Estimate the forward-shock radius from the steepest pressure drop."""

    coordinate = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    pressure = (
        rout.fluid.rho
        / (rout.fluid.mu * unyt.mp)
        * unyt.kb
        * rout.fluid.temp
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
):
    """Estimate the cavity-side shell edge from the innermost density crossing."""

    coordinate = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    density = rout.fluid.rho

    coordinate_values = coordinate.to_value(coordinate.units)
    density_values = density.to_value(density.units)
    mask = coordinate_values >= 0.0
    coordinate_values = coordinate_values[mask]
    density_values = density_values[mask]
    coordinate = coordinate[mask]

    if density_values.size < 2:
        return None

    threshold = (
        ambient_density.to_value(density.units)
        * float(np.asarray(threshold_factor).reshape(-1)[0])
    )
    above = density_values >= threshold
    if not np.any(above):
        return None

    starts = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    if starts.size == 0:
        return None

    if above[0]:
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


def weaver_forward_shock_radius(rout, icparams, runparams):
    """Return the Weaver shock radius for a loaded snapshot."""

    return wa.shock_radius(
        rout.par.time,
        icparams['rhoini'],
        runparams['rho_outflow'],
        runparams['vel_outflow'],
        icparams['rinj'],
    )


def plot_density_snapshot(ax, rout, **kwargs):
    """Plot one density snapshot on a supplied axis."""

    plt.sca(ax)
    rplot1d(rout, yquan='rho', showhalf=0, showfig=0, **kwargs)
    ax.set_yscale('log')


def plot_temperature_snapshot(ax, rout, **kwargs):
    """Plot one temperature snapshot on a supplied axis."""

    plt.sca(ax)
    rplot1d(rout, yquan='temp', showhalf=0, showfig=0, **kwargs)
    ax.set_yscale('log')


def plot_profile_snapshot(ax, rout, yquan, xunit=unyt.pc, **kwargs):
    """Plot one radial profile on a supplied axis using ``xunit``."""

    coordinate = 0.5 * (rout.mesh.boundary[1:] + rout.mesh.boundary[:-1])
    coordinate_values = coordinate.to_value(xunit)
    nonnegative = coordinate_values >= 0.0
    ax.plot(
        coordinate_values[nonnegative],
        getattr(rout.fluid, yquan).to_value(getattr(rout.fluid, yquan).units)[nonnegative],
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
            'rho',
            ls='none',
            marker='o',
            mfc='none',
            markevery=5,
            color=color,
        )
        plot_profile_snapshot(
            ax_temperature,
            rout,
            'temp',
            ls='none',
            marker='o',
            mfc='none',
            markevery=5,
            color=color,
        )
        if rout.par.time > 0 * rout.par.time.units:
            shock_radius = weaver_forward_shock_radius(rout, icparams, runparams)
            shock_value = shock_radius.to_value(icparams['rinj'].units).item()
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
    shell_threshold_factor = runparams['shell_edge_density_threshold_factor']

    for rout in snapshots:
        if rout.par.time <= 0 * rout.par.time.units:
            continue
        numerical_radius = shell_inner_edge_radius(
            rout,
            icparams['rhoini'],
            shell_threshold_factor,
        )
        if numerical_radius is None:
            continue
        weaver_radius = weaver_forward_shock_radius(rout, icparams, runparams)
        time_myr = rout.par.time.to_value(unyt.Myr)
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


def ReadandPlot(outfilename, icparams, runparams, **kwargs):
    rout = load_snapshot(outfilename, icparams)
    plot_density_snapshot(plt.gca(), rout, **kwargs)
    if np.all(rout.par.time > 0 * rout.par.time.units):
        shock_radius = weaver_forward_shock_radius(rout, icparams, runparams)
        plt.axvline(
            x=shock_radius.to_value(icparams['rinj'].units).item(),
            color=kwargs['color'],
            ls='dashed',
        )
