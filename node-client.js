const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const protobuf = require("protobufjs");

const options = {
  includeDirs: ["."],
  oneofs: true,
};

const grpcApi = protoLoader.loadSync("modal_proto/api.proto", options);
const ModalClient = grpc.loadPackageDefinition(grpcApi).modal.client.ModalClient;
const pbApi = protobuf.loadSync("modal_proto/api.proto");
const PayloadValue = pbApi.lookupType("modal.client.PayloadValue");

const client = new ModalClient(
  "api.modal.com",
  grpc.credentials.createSsl()
);

const metadata = new grpc.Metadata();
metadata.add("x-modal-token-id", process.env.MODAL_TOKEN_ID);
metadata.add("x-modal-token-secret", process.env.MODAL_TOKEN_SECRET);
metadata.add("x-modal-client-version", "0.64.0");
metadata.add("x-modal-client-type", "1");

function serialize(data) {
  if (typeof data == "string" ) {
    return { type: 1, strValue: data };
  } else if (typeof data == "number") {
    if (Number.isInteger(data)) {
      return { type: 2, intValue: data };
    } else {
      return { type: 4, floatValue: data };
    }
  } else if (typeof data == "boolean") {
    return { type: 3, boolValue: data };
  } else if (Array.isArray(data)) {
    const items = data.map(serialize);
    return { type: 6, listValue: { items }};
  } else if (typeof data == "object") {
    const keys = Object.keys(data);
    const values = keys.map(key => serialize(data[key]));
    return { type: 7, dictValue: { keys, values } };
  } else {
    throw Exception(`can't serialize ${typeof data}`);
  }
}

function deserialize(pv) {
  if (pv.type == 1) {
    return pv.strValue;
  } else if (pv.type == 2) {
    return pv.intValue;
  } else if (pv.type == 3) {
    return pv.boolValue;
  } else if (pv.type == 4) {
    return pv.floatValue;
  } else if (pv.type == 6) {
    return pv.listValue.items.map(deserialize);
  } else {
    throw Exception(`can't deserialize ${pv.type}`)
  }
}

function createFunction(functionId) {
  return async function call() {
    // convert arguments to a list
    const args = [];
    for (let i = 0; i < arguments.length; i++) {
      args.push(arguments[i]);
    }

    // serialize payload
    const kwargs = {};
    const inputMessage = serialize([args, kwargs]);
    const inputSerialized = PayloadValue.encode(inputMessage).finish();

    // construct function map input
    const input = {
      idx: 0,
      input: {
	finalInput: true,
	dataFormat: 4,
	args: inputSerialized,
      }
    };
    const functionMapReq = {
      functionId: functionId,
      functionCallType: 1,
      pipelinedInputs: [input],
      functionCallInvocationType: 1,
    };

    // Enqueue input
    const mapResp = await new Promise((resolve, reject) => {
      client.FunctionMap(functionMapReq, metadata, (error, ret) => {
	if (error) {
	  reject(error);
	} else {
	  resolve(ret);
	}
      });
    });
    const functionCallId = mapResp.functionCallId;

    // Dequeue output
    const getReq = {
      functionCallId,
      lastEntryId: "0-0",
      timeout: 60,
    };
    const getResp = await new Promise((resolve, reject) => {
      client.FunctionGetOutputs(getReq, metadata, (error, ret) => {
	if (error) {
	  reject(error);
	} else {
	  resolve(ret);
	}
      });
    });
    const outputSerialized = getResp.outputs[0].result.data;

    // Deserialize data
    const outputMessage = PayloadValue.decode(outputSerialized);
    return deserialize(outputMessage);
  }
}

async function getFunction(appName, functionName) {
  const req = {
    appName: appName,
    objectTag: functionName,
    namespace: 1,
    environmentName: "main",
  };
  return new Promise((resolve, reject) => {
    client.FunctionGet(req, metadata, (error, ret) => {
      if (error) {
        reject(error);
      } else {
        const f = createFunction(ret.functionId);
        resolve(f);
      }
    });
  });
}

async function run() {
  const f = await getFunction("payload-value", "f");
  const ret = await f(123);
  console.log(ret);
}

run();
