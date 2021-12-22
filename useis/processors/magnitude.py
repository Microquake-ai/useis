from ..core.project_manager import ProjectManager
from uquake.core.stream import Stream
import numpy as np


class Magnitude(ProjectManager):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str):
        """
        Based object for the magnitude calculation
        :param base_projects_path: base project path
        :type base_projects_path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        """

        super().__init__(base_projects_path, project_name, network_code)

        self.files.magnitude_settings = self.paths.config / \
                                        'magnitude_settings.toml'

        if not self.files.magnitude_settings.is_file():
            settings_template = Path(os.path.realpath(__file__)).parent / \
                                '../settings/magnitude_settings_template.toml'

            shutil.copyfile(settings_template,
                            self.files.magnitude_settings)

            super().__init__(base_projects_path, project_name, network_code)
            
        


class MwPE(Magnitude):
    def __init__(self, base_projects_path: Path, project_name: str,
                 network_code: str):
        """
        Based object for the magnitude calculation
        :param base_projects_path: base project path
        :type base_projects_path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        """

        super().__init__(base_projects_path, project_name, network_code)

    def predict(self, st: Stream, event_location: np.array,
                use_distance_along_rays=True):
        """

        :param st: waveform
        :type st: :py:class:uquake.core.stream.Stream:
        :param event_location: event location
        :type event_location: :py:class:numpy.array:
        :param use_distance_along_rays: Use the distance along the rays if True
        use the straight line distance if False. Default True.
        :type use_distance_along_rays: bool
        :return: the magnitude value estimate
        :rtype: float
        """

        a = self.settings.MwPE.a
        b = self.settings.MwPE.c

        min_frequency = 0
        for tr in st:
            trace_time = (tr.stats.endtime - tr.stats.starttime) * 0.9
            if 1 / trace_time > min_frequency:
                min_frequency = 1 / trace_time

        min_frequency_1 = min_frequency
        min_frequency_2 = 10 ** (np.log10(min_frequency) + 0.2)

        pre_filt = [min_frequency_1, min_frequency_2]

        max_frequency_2 = self.settings.MwPE.max_frequency
        max_frequency_1 = 10 ** (np.log10(max_frequency_2) - 0.2)

        pre_filt = [min_frequency_1, min_frequency_2,
                    max_frequency_1, max_frequency_2]

        water_level = self.settings.MwPE.water_level

        st.attach_response(self.inventory)
        st.remove_response(inventory=self.inventory, pre_filt=pre_filt,
                           water_level=water_level, output='VEL')

        if use_distance_along_rays:
            rays = self.travel_times.ray_tracer(event_location,
                                                seed_labels='current_event')

        magnitudes = []
        for network in self.inventory:
            for station in network:
                st2 = st.copy().select(network=network, station=station)
                if len(st2) == 1:
                    amplitude = np.max(st2[0].data) * np.sqrt(3)
                else:
                    amplitude = np.max(st2.composite()[0].data)

        networks = self.inventory

            
            


