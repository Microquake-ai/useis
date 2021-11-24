from uquake.waveform.amp_measures import calc_velocity_flux
from uquake.waveform.mag import calculate_energy_from_flux

# from uquake.processors.processing_unit import ProcessingUnit

# from uquake.core.focal_mechanism import calc

from uquake.waveform.simple_mag import moment_magnitude
from uquake.waveform.amp_measures import measure_pick_amps
from ..core.project_manager import ProjectManager
from uquake.core.stream import Stream
from uquake.core.event import Catalog


class Magnitude(ProjectManager):

    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str, use_srces: bool = False):
        """
        Object to control NLLoc execution and manage the required grids, inputs
        and outputs
        :param base_projects_path:
        :param project_name:
        :param network_code:
        :param use_srces:
        """

        super().__init__(base_projects_path, project_name, network_code,
                         use_srces=use_srces)

    def calculate_magnitude(self, stream: Stream, cat: Catalog):

        return moment_magnitude(stream, cat, self.inventory,
                                self.p_velocity, self.s_velocity)


    #
    #
    #
    # def process(
    #     self,
    #     **kwargs
    # ):
    #
    #     stream = kwargs['stream']
    #     cat = kwargs['cat']
    #
    #     inventory = self.settings.inventory
    #     stream.attach_response(inventory)
    #
    #     vp, vs = velocity.get_velocities()
    #     cat_moment = moment_magnitude(stream, cat, inventory, vp, vs)
    #
    #     cat_flux = calc_velocity_flux(stream, cat_moment, inventory,
    #                                   phase_list=['P', 'S'])
    #     cat_energy = calculate_energy_from_flux(cat_flux, inventory, vp, vs,
    #                                             use_sdr_rad=False)
    #
    #     mag = cat_energy[0].magnitudes[-1]
    #     energy_p = mag.energy_p_joule
    #     energy_s = mag.energy_s_joule
    #     energy = mag.energy_joule
    #     cat_energy[0].preferred_magnitude().energy_p_joule = energy_p
    #     cat_energy[0].preferred_magnitude().energy_s_joule = energy_s
    #     cat_energy[0].preferred_magnitude().energy_joule = energy
    #
    #     return cat_energy

