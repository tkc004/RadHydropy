"""Shell-orbit event tracking for the Bertschinger splashback study."""

import numpy as np


class ShellOrbitTracker:
    """Track first turnaround, pericentre, apocentre, and accretion events."""

    def __init__(self, initial_q, cosmology, recent_window_fraction=0.5):
        self.initial_q = np.asarray(initial_q, dtype=float)
        self.cosmology = cosmology
        self.recent_window_fraction = float(recent_window_fraction)
        self.events = {
            int(index): {'turnaround': None, 'pericentre': None,
                         'apocentre': None, 'accretion': None}
            for index in range(self.initial_q.size)
        }
        self.previous = None
        self.latest_time = None

    def _r200m(self, radius_comoving_code, mass_comoving_code, cosmic_time):
        order = np.argsort(radius_comoving_code)
        radius_comoving_code = np.asarray(radius_comoving_code)[order]
        mass_comoving_code = np.asarray(mass_comoving_code)[order]
        mean_density_comoving_code = np.cumsum(mass_comoving_code) / (
            4.0 * np.pi / 3.0 * radius_comoving_code**3
        )
        target = 200.0 * float(self.cosmology.background_density(cosmic_time))
        crossing = np.flatnonzero(
            (mean_density_comoving_code[:-1] >= target)
            & (mean_density_comoving_code[1:] < target))
        if not crossing.size:
            return None
        index = int(crossing[-1])
        return float(np.exp(np.interp(
            np.log(target),
            np.log(mean_density_comoving_code[index:index + 2][::-1]),
            np.log(radius_comoving_code[index:index + 2][::-1]))))

    def observe(self, cosmic_time, scale_factor, radius_comoving_code,
                vel_supercomoving_code, mass_comoving_code,
                shell_id):
        """Consume one accepted solver state, preserving shell identities."""
        if shell_id is None:
            raise ValueError('ShellOrbitTracker requires persistent shell IDs')
        ids = np.asarray(shell_id, dtype=int)
        radius_comoving_code = np.asarray(radius_comoving_code, dtype=float)
        vel_supercomoving_code = np.asarray(vel_supercomoving_code, dtype=float)
        mass_comoving_code = np.asarray(mass_comoving_code, dtype=float)
        radius_proper_code = float(scale_factor) * radius_comoving_code
        vel_proper_code = (
            float(self.cosmology.hubble(cosmic_time)) * radius_proper_code +
            vel_supercomoving_code / float(scale_factor))
        r200m = self._r200m(radius_comoving_code, mass_comoving_code, cosmic_time)
        current = {int(i): (radius_proper_code[j], vel_proper_code[j])
                   for j, i in enumerate(ids)}
        if self.previous is not None:
            previous_time, previous, previous_r200m = self.previous
            for shell_index, (r_now, v_now) in current.items():
                if shell_index not in previous:
                    continue
                r_old, v_old = previous[shell_index]
                event_time = float(cosmic_time)
                event_radius = float(r_now)
                if v_old * v_now < 0.0:
                    fraction = -v_old / (v_now - v_old)
                    event_time = float(previous_time + fraction *
                                       (cosmic_time - previous_time))
                    event_radius = float(r_old + fraction * (r_now - r_old))
                event = self.events[shell_index]
                if v_old >= 0.0 and v_now < 0.0:
                    if event['turnaround'] is None:
                        event['turnaround'] = (event_time, event_radius)
                    elif (event['pericentre'] is not None and
                          event['apocentre'] is None):
                        event['apocentre'] = (event_time, event_radius)
                elif v_old <= 0.0 and v_now > 0.0:
                    if (event['turnaround'] is not None and
                            event['pericentre'] is None):
                        event['pericentre'] = (event_time, event_radius)
                if (event['accretion'] is None and previous_r200m is not None
                        and r200m is not None and v_now < 0.0
                        and r_old >= previous_r200m and r_now < r200m):
                    fraction = (previous_r200m - r_old) / (
                        (r_now - r_old) - (r200m - previous_r200m))
                    event['accretion'] = (
                        float(previous_time + fraction *
                              (cosmic_time - previous_time)),
                        float(r_old + fraction * (r_now - r_old)))
        self.previous = (float(cosmic_time), current, r200m)
        self.latest_time = float(cosmic_time)
        return r200m

    def recent_first_apocenters(self, cosmic_time):
        """Return first-apocentre radii for recently accreted shells."""
        threshold = float(cosmic_time) * (1.0 - self.recent_window_fraction)
        values = []
        for event in self.events.values():
            if event['accretion'] is None or event['apocentre'] is None:
                continue
            if event['accretion'][0] >= threshold:
                values.append(event['apocentre'][1])
        return np.asarray(values, dtype=float)

    def first_apocenter_events(self):
        """Return ``(time_cosmic, radius_proper, accretion_time)`` for events."""
        events = []
        for event in self.events.values():
            if event['apocentre'] is None:
                continue
            accretion = (np.nan if event['accretion'] is None
                         else event['accretion'][0])
            events.append((event['apocentre'][0], event['apocentre'][1],
                           accretion))
        return np.asarray(events, dtype=float).reshape((-1, 3))
