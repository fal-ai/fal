# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from isolate_proto import controller_pb2 as controller__pb2


class IsolateControllerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Run = channel.unary_stream(
                '/controller.IsolateController/Run',
                request_serializer=controller__pb2.HostedRun.SerializeToString,
                response_deserializer=controller__pb2.HostedRunResult.FromString,
                )
        self.Map = channel.unary_stream(
                '/controller.IsolateController/Map',
                request_serializer=controller__pb2.HostedMap.SerializeToString,
                response_deserializer=controller__pb2.HostedRunResult.FromString,
                )
        self.CreateUserKey = channel.unary_unary(
                '/controller.IsolateController/CreateUserKey',
                request_serializer=controller__pb2.CreateUserKeyRequest.SerializeToString,
                response_deserializer=controller__pb2.CreateUserKeyResponse.FromString,
                )
        self.ListUserKeys = channel.unary_unary(
                '/controller.IsolateController/ListUserKeys',
                request_serializer=controller__pb2.ListUserKeysRequest.SerializeToString,
                response_deserializer=controller__pb2.ListUserKeysResponse.FromString,
                )
        self.RevokeUserKey = channel.unary_unary(
                '/controller.IsolateController/RevokeUserKey',
                request_serializer=controller__pb2.RevokeUserKeyRequest.SerializeToString,
                response_deserializer=controller__pb2.RevokeUserKeyResponse.FromString,
                )
        self.RegisterApplication = channel.unary_stream(
                '/controller.IsolateController/RegisterApplication',
                request_serializer=controller__pb2.RegisterApplicationRequest.SerializeToString,
                response_deserializer=controller__pb2.RegisterApplicationResult.FromString,
                )
        self.UpdateApplication = channel.unary_unary(
                '/controller.IsolateController/UpdateApplication',
                request_serializer=controller__pb2.UpdateApplicationRequest.SerializeToString,
                response_deserializer=controller__pb2.UpdateApplicationResult.FromString,
                )
        self.ListApplications = channel.unary_unary(
                '/controller.IsolateController/ListApplications',
                request_serializer=controller__pb2.ListApplicationsRequest.SerializeToString,
                response_deserializer=controller__pb2.ListApplicationsResult.FromString,
                )
        self.DeleteApplication = channel.unary_unary(
                '/controller.IsolateController/DeleteApplication',
                request_serializer=controller__pb2.DeleteApplicationRequest.SerializeToString,
                response_deserializer=controller__pb2.DeleteApplicationResult.FromString,
                )
        self.SetAlias = channel.unary_unary(
                '/controller.IsolateController/SetAlias',
                request_serializer=controller__pb2.SetAliasRequest.SerializeToString,
                response_deserializer=controller__pb2.SetAliasResult.FromString,
                )
        self.DeleteAlias = channel.unary_unary(
                '/controller.IsolateController/DeleteAlias',
                request_serializer=controller__pb2.DeleteAliasRequest.SerializeToString,
                response_deserializer=controller__pb2.DeleteAliasResult.FromString,
                )
        self.ListAliases = channel.unary_unary(
                '/controller.IsolateController/ListAliases',
                request_serializer=controller__pb2.ListAliasesRequest.SerializeToString,
                response_deserializer=controller__pb2.ListAliasesResult.FromString,
                )
        self.SetSecret = channel.unary_unary(
                '/controller.IsolateController/SetSecret',
                request_serializer=controller__pb2.SetSecretRequest.SerializeToString,
                response_deserializer=controller__pb2.SetSecretResponse.FromString,
                )
        self.ListSecrets = channel.unary_unary(
                '/controller.IsolateController/ListSecrets',
                request_serializer=controller__pb2.ListSecretsRequest.SerializeToString,
                response_deserializer=controller__pb2.ListSecretsResponse.FromString,
                )
        self.ListAliasRunners = channel.unary_unary(
                '/controller.IsolateController/ListAliasRunners',
                request_serializer=controller__pb2.ListAliasRunnersRequest.SerializeToString,
                response_deserializer=controller__pb2.ListAliasRunnersResponse.FromString,
                )


class IsolateControllerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Run(self, request, context):
        """Run the given function on the specified environment. Streams logs
        and the result originating from that function.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Map(self, request, context):
        """Run the given function in parallel with the given inputs
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateUserKey(self, request, context):
        """Creates an authentication key for a user
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListUserKeys(self, request, context):
        """Lists the user's authentication keys
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RevokeUserKey(self, request, context):
        """Revokes an authentication key for a user
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterApplication(self, request, context):
        """Register a funtion
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateApplication(self, request, context):
        """Update configuration of an existing application.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListApplications(self, request, context):
        """List functions
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteApplication(self, request, context):
        """Delete a function
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAlias(self, request, context):
        """Set alias to point to an existing application.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteAlias(self, request, context):
        """Delete an alias.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListAliases(self, request, context):
        """List aliased registered functions
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetSecret(self, request, context):
        """Sets a user secret.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListSecrets(self, request, context):
        """Lists all secrets
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListAliasRunners(self, request, context):
        """List alias runners in detail
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_IsolateControllerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Run': grpc.unary_stream_rpc_method_handler(
                    servicer.Run,
                    request_deserializer=controller__pb2.HostedRun.FromString,
                    response_serializer=controller__pb2.HostedRunResult.SerializeToString,
            ),
            'Map': grpc.unary_stream_rpc_method_handler(
                    servicer.Map,
                    request_deserializer=controller__pb2.HostedMap.FromString,
                    response_serializer=controller__pb2.HostedRunResult.SerializeToString,
            ),
            'CreateUserKey': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateUserKey,
                    request_deserializer=controller__pb2.CreateUserKeyRequest.FromString,
                    response_serializer=controller__pb2.CreateUserKeyResponse.SerializeToString,
            ),
            'ListUserKeys': grpc.unary_unary_rpc_method_handler(
                    servicer.ListUserKeys,
                    request_deserializer=controller__pb2.ListUserKeysRequest.FromString,
                    response_serializer=controller__pb2.ListUserKeysResponse.SerializeToString,
            ),
            'RevokeUserKey': grpc.unary_unary_rpc_method_handler(
                    servicer.RevokeUserKey,
                    request_deserializer=controller__pb2.RevokeUserKeyRequest.FromString,
                    response_serializer=controller__pb2.RevokeUserKeyResponse.SerializeToString,
            ),
            'RegisterApplication': grpc.unary_stream_rpc_method_handler(
                    servicer.RegisterApplication,
                    request_deserializer=controller__pb2.RegisterApplicationRequest.FromString,
                    response_serializer=controller__pb2.RegisterApplicationResult.SerializeToString,
            ),
            'UpdateApplication': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateApplication,
                    request_deserializer=controller__pb2.UpdateApplicationRequest.FromString,
                    response_serializer=controller__pb2.UpdateApplicationResult.SerializeToString,
            ),
            'ListApplications': grpc.unary_unary_rpc_method_handler(
                    servicer.ListApplications,
                    request_deserializer=controller__pb2.ListApplicationsRequest.FromString,
                    response_serializer=controller__pb2.ListApplicationsResult.SerializeToString,
            ),
            'DeleteApplication': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteApplication,
                    request_deserializer=controller__pb2.DeleteApplicationRequest.FromString,
                    response_serializer=controller__pb2.DeleteApplicationResult.SerializeToString,
            ),
            'SetAlias': grpc.unary_unary_rpc_method_handler(
                    servicer.SetAlias,
                    request_deserializer=controller__pb2.SetAliasRequest.FromString,
                    response_serializer=controller__pb2.SetAliasResult.SerializeToString,
            ),
            'DeleteAlias': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteAlias,
                    request_deserializer=controller__pb2.DeleteAliasRequest.FromString,
                    response_serializer=controller__pb2.DeleteAliasResult.SerializeToString,
            ),
            'ListAliases': grpc.unary_unary_rpc_method_handler(
                    servicer.ListAliases,
                    request_deserializer=controller__pb2.ListAliasesRequest.FromString,
                    response_serializer=controller__pb2.ListAliasesResult.SerializeToString,
            ),
            'SetSecret': grpc.unary_unary_rpc_method_handler(
                    servicer.SetSecret,
                    request_deserializer=controller__pb2.SetSecretRequest.FromString,
                    response_serializer=controller__pb2.SetSecretResponse.SerializeToString,
            ),
            'ListSecrets': grpc.unary_unary_rpc_method_handler(
                    servicer.ListSecrets,
                    request_deserializer=controller__pb2.ListSecretsRequest.FromString,
                    response_serializer=controller__pb2.ListSecretsResponse.SerializeToString,
            ),
            'ListAliasRunners': grpc.unary_unary_rpc_method_handler(
                    servicer.ListAliasRunners,
                    request_deserializer=controller__pb2.ListAliasRunnersRequest.FromString,
                    response_serializer=controller__pb2.ListAliasRunnersResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'controller.IsolateController', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class IsolateController(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Run(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/controller.IsolateController/Run',
            controller__pb2.HostedRun.SerializeToString,
            controller__pb2.HostedRunResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Map(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/controller.IsolateController/Map',
            controller__pb2.HostedMap.SerializeToString,
            controller__pb2.HostedRunResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CreateUserKey(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/CreateUserKey',
            controller__pb2.CreateUserKeyRequest.SerializeToString,
            controller__pb2.CreateUserKeyResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListUserKeys(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/ListUserKeys',
            controller__pb2.ListUserKeysRequest.SerializeToString,
            controller__pb2.ListUserKeysResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RevokeUserKey(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/RevokeUserKey',
            controller__pb2.RevokeUserKeyRequest.SerializeToString,
            controller__pb2.RevokeUserKeyResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterApplication(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/controller.IsolateController/RegisterApplication',
            controller__pb2.RegisterApplicationRequest.SerializeToString,
            controller__pb2.RegisterApplicationResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateApplication(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/UpdateApplication',
            controller__pb2.UpdateApplicationRequest.SerializeToString,
            controller__pb2.UpdateApplicationResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListApplications(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/ListApplications',
            controller__pb2.ListApplicationsRequest.SerializeToString,
            controller__pb2.ListApplicationsResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteApplication(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/DeleteApplication',
            controller__pb2.DeleteApplicationRequest.SerializeToString,
            controller__pb2.DeleteApplicationResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAlias(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/SetAlias',
            controller__pb2.SetAliasRequest.SerializeToString,
            controller__pb2.SetAliasResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteAlias(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/DeleteAlias',
            controller__pb2.DeleteAliasRequest.SerializeToString,
            controller__pb2.DeleteAliasResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListAliases(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/ListAliases',
            controller__pb2.ListAliasesRequest.SerializeToString,
            controller__pb2.ListAliasesResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetSecret(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/SetSecret',
            controller__pb2.SetSecretRequest.SerializeToString,
            controller__pb2.SetSecretResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListSecrets(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/ListSecrets',
            controller__pb2.ListSecretsRequest.SerializeToString,
            controller__pb2.ListSecretsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListAliasRunners(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/controller.IsolateController/ListAliasRunners',
            controller__pb2.ListAliasRunnersRequest.SerializeToString,
            controller__pb2.ListAliasRunnersResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
