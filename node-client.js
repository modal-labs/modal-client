const grpc = require("@grpc/grpc-js");
var protoLoader = require("@grpc/proto-loader");

const options = {
 includeDirs: ["."],
};

var packageDefinition = protoLoader.loadSync("modal_proto/api.proto", options);
const ModalClient = grpc.loadPackageDefinition(packageDefinition).modal.client.ModalClient;

const client = new ModalClient(
  "api.modal.com",
  grpc.credentials.createSsl()
);

client.ClientHello({}, (error, ret) => {
  if (error) throw error
  console.log(ret);
});
