"""Helpers reused from the maintained 20 pc Stromgren example."""

import importlib.util
import sys
from pathlib import Path


TEMPLATE_TOOLS = (
    Path(__file__).resolve().parents[1]
    / "DynamicStromgrenSpherePhotoheating20pc1D"
    / "tools.py"
)
spec = importlib.util.spec_from_file_location(
    "_radhydropy_dynamic_stromgren_20pc_tools",
    TEMPLATE_TOOLS,
)
template = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = template
spec.loader.exec_module(template)

for name, value in vars(template).items():
    if not name.startswith("__"):
        globals()[name] = value


def save_plot(mesh, fluid, par, config, figure_filename):
    """Save the inherited profile plot with a linear velocity axis."""
    interior = interior_slice(par)
    radius_pc = _to_kpc(mesh.coordinate[interior], par) * (1.0 * unyt.kpc).to_value(unyt.pc)
    number_density = _to_number_density(fluid.rho_code[interior], par)
    velocity = _to_km_s(fluid.vel_code[interior], par)
    neutral_fraction = np.asarray(fluid.xHI[interior], dtype=float)
    pressure = _to_pressure(fluid.pre_code[interior], par)
    temperature = _to_temperature(fluid.temp_code[interior], par)
    plot_radius_max = config['plot_radius_max'].to_value(unyt.pc)
    radius_unit = config.get('reference_radius_unit', 15.0 * unyt.kpc)
    density_reference = load_reference_profile(
        config.get('density_reference_filename', None), radius_unit, log_value=True
    )
    velocity_reference = load_reference_profile(
        config.get('velocity_reference_filename', None), radius_unit, log_value=False
    )
    pressure_reference = load_reference_profile(
        config.get('pressure_reference_filename', None), radius_unit, log_value=True
    )
    neutral_fraction_reference = load_reference_profile(
        config.get('neutral_fraction_reference_filename', None), radius_unit, log_value=True
    )
    reference_radius_scale = (1.0 * unyt.kpc).to_value(unyt.pc)
    for reference in (
        density_reference,
        velocity_reference,
        pressure_reference,
        neutral_fraction_reference,
    ):
        if reference is not None:
            reference['radius_kpc'] *= reference_radius_scale

    fig, axes = plt.subplots(5, 1, figsize=(7.4, 11.0), sharex=True)
    axes[0].plot(radius_pc, number_density, color='tab:blue', lw=1.8, label='RadHydropy')
    scatter_reference(axes[0], density_reference)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$n$ [cm$^{-3}$]')
    axes[0].legend(frameon=False, loc='best')

    axes[1].plot(radius_pc, velocity, color='tab:orange', lw=1.8, label='RadHydropy')
    scatter_reference(axes[1], velocity_reference)
    axes[1].set_yscale('linear')
    axes[1].set_ylabel(r'$v_r$ [km s$^{-1}$]')
    axes[1].legend(frameon=False, loc='best')

    axes[2].plot(
        radius_pc,
        np.clip(neutral_fraction, 1.0e-8, 1.0),
        color='tab:green',
        lw=1.8,
        label='RadHydropy',
    )
    scatter_reference(axes[2], neutral_fraction_reference)
    axes[2].set_yscale('log')
    axes[2].set_ylabel(r'$x_{\rm HI}$')
    axes[2].legend(frameon=False, loc='best')

    axes[3].plot(radius_pc, pressure, color='tab:red', lw=1.8, label='RadHydropy')
    scatter_reference(axes[3], pressure_reference)
    axes[3].set_yscale('log')
    axes[3].set_ylabel(r'$P$ [g cm$^{-1}$ s$^{-2}$]')
    axes[3].legend(frameon=False, loc='best')

    axes[4].plot(radius_pc, temperature, color='tab:purple', lw=1.8, label='RadHydropy')
    axes[4].set_yscale('log')
    axes[4].set_ylabel(r'$T$ [K]')
    axes[4].set_xlabel('Radius [pc]')
    axes[4].legend(frameon=False, loc='best')

    for ax in axes:
        ax.set_xlim(0.0, plot_radius_max)
        ax.grid(True, which='both', alpha=0.25)
    final_time_myr = config['timesim'].to_value(unyt.Myr)
    fig.suptitle('Dynamic photoheated Stromgren sphere at %.3g Myr' % final_time_myr)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
