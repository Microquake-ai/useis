from useis.services import grid_service_pb2_grpc as gs_grpc
from useis.processors.nlloc import NLLOC
from useis.settings.settings import Settings
from useis.services.grid_service_pb2 import GenericMessage
from useis.services.nlloc import VelocityGrid3D
import grpc
from concurrent import futures
from loguru import logger
from uquake.core.inventory import Inventory

settings = Settings


class GridService(gs_grpc.GridServicer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_velocity_grid(self, request, context):
        project = request.project
        network = request.network_code
        nll = NLLOC('projects', project, network)
        velocity = VelocityGrid3D.from_proto(request)
        try:
            nll.add_velocity(velocity)
        except Exception as e:
            logger.error(e)

        return GenericMessage(message='success')

    def add_inventory(self, request_iterator, context):
        file_bytes = []
        for request in request_iterator:
            file_bytes.append(request.file_bytes)
        inventory = Inventory.from_bytes(file_bytes)

    # def add_inventory(self, request, context):
    #     project = request.project
    #     network = request.network_code
    #     inventory = Inventory.from_bytes(request.file_bytes)

    def add_srces(self, request, context):
        project = request.project
        network = request.network
        sites_proto = request.sites

        return 'allo'


class Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gs_grpc.add_GridServicer_to_server(GridService(), server)
    server.add_insecure_port('[::]:50051')

    def start(self):
        logger.info('starting server')
        self.server.start()
        logger.info('server started')

    def stop(self):
        logger.info('stopping server')
        self.server.stop()
        logger.info('server stopped')


def serve():
    server = Server()
    server.start()
    server.server.wait_for_termination()


if __name__ == '__main__':
    serve()

