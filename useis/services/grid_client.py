import grpc
import uquake.grid.nlloc

import useis.services.grid_service_pb2_grpc as pb2_grpc
import useis.services.grid_service_pb2 as pb2
from useis.services.nlloc import VelocityGrid3D, Srces

from uquake.core import inventory


class GridClient(object):
    """
    Client for gRPC functionality
    """

    def __init__(self, host='localhost', port=50051):
        self.host = host
        self.server_port = port

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = pb2_grpc.GridStub(self.channel)

    def add_velocity_grid_3d(self, velocity_grid_3d:
        uquake.grid.nlloc.VelocityGrid3D, project):
        """
        Client function to call the rpc add_velocity_grid_3d to add a grid to
        the project
        """

        velocity_grid = VelocityGrid3D.from_velocity_grid_3d(velocity_grid_3d)
        velocity_proto = velocity_grid.to_proto()
        velocity_proto.project = project
        return self.stub.add_velocity_grid(velocity_proto)

    def add_inventory(self, inv, project, network):
        """
        :param inv: uquake.core.inventory.Inventory
        :param project:
        :param network:
        :return:
        """

        file_bytes = inv.to_bytes()
        inventory_proto = pb2.GenericFile()
        # inventory_proto.project = project
        # inventory_proto.network_code = network
        for file_byte in file_bytes:
            inventory_proto.file_bytes = bytes(file_byte)
            self.stub.add_inventory(inventory_proto)

        # inventory_proto = pb2.GenericFile(file_bytes=bytes(file_bytes),
        #                                   project=project,
        #                                   network_code=network)
        # return self.stub.add_inventory(inventory_proto)

    def add_srces(self, srces, project, network):
        """
        :param srces:
        :type srces: uquake.nlloc.nlloc.Srces
        :param project:
        :param network:
        :return:
        """
        pb_srces = Srces.from_srces(srces)
        sites_proto = pb_srces.to_proto()

        pb_srces = pb2.Srces()
        pb_srces.project = project
        pb_srces.network = network
        pb_srces.sites.extend(sites_proto)

        self.stub.add_srces(pb_srces)












if __name__ == '__main__':
    client = GridClient()
    result = client.add_velocity_grid_3d()
    result = client.get_url(message="Hello Server you there?")
    print(f'{result}')