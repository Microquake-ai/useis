from .grid_service_pb2_grpc import GridServicer as GS
from ..processors.nlloc import NLLOC
from ..settings.settings import Settings
from .grid_service_pb2 import VelocityGrid3D, GenericMessage
import gr

settings = Settings
nll = NLLOC()


class GridServicer(GS):
    def AddVelocityGrid(self, request, context):
        project = request.project
        network = request.network

        nll = NLLOC('projects', project, network)

        velocity = VelocityGrid3D.from_proto(request)

        nll.add_velocities(velocity)

        return GenericMessage(message='completed')


