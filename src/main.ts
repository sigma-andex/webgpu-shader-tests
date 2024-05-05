import "./style.css";
import attnKWeights from "./blk.0.attn_k.weight.bin?arraybuffer";
import attnKWeightsF32 from "./blk.0.attn_k.weight.f32.bin?arraybuffer";

async function main() {
  // tensor=blk.0.attn_k.weight,shape=[256, 256]
  const input = new Uint8Array(attnKWeights);
  const f32Input = new Float32Array(attnKWeightsF32);
  console.log(f32Input);

  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
  if (!device) {
    console.error("need a browser that supports WebGPU");
    return;
  } else {
    console.info(`Got WebGPU device`);
  }

  const module = device.createShaderModule({
    label: "dequant_q4k compute module",
    code: `
    @group(0) @binding(0)
    var<storage, read_write> ds: array<f32>;

    // super-blocks scale for quantized mins
    @group(0) @binding(1)
    var<storage, read_write> dmins: array<f32>;

    // scales and mins, quantized with 6 bits
    @group(0) @binding(2)
    var<storage, read_write> scales: array<u32>;

    // 4--bit quants
    @group(0) @binding(3)
    var<storage, read_write> qs: array<u32>;

    // output buffer containing f32 dequantised values
    @group(0) @binding(4)
    var<storage, read_write> dequantised: array<f32>;

    struct Debug {
      global_id: vec3u,
      local_id: vec3u
    }
    @group(0) @binding(5)
    var<storage, read_write> debug: Debug;
    
    
    @compute @workgroup_size(1) fn dequant_q4k(
      @builtin(global_invocation_id) global_id: vec3u,
      @builtin(local_invocation_id) local_id: vec3u) {

      debug.global_id = global_id;
      debug.local_id = local_id;

      let ix = 0;
      let d = ds[ix];
      dequantised[0] = d;

      
    }
    `,
  });

  const layout = device.createBindGroupLayout({
    label: `bindGroupLayout for work buffer`,
    entries: [...Array(6).keys()].map((i) => ({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "storage",
      },
    })),
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "dequant_q4k pipeline layout",
    bindGroupLayouts: [layout],
  });
  const pipeline = device.createComputePipeline({
    label: "dequant_q4k compute pipeline",
    layout: pipelineLayout,
    compute: {
      module,
      entryPoint: "dequant_q4k",
    },
  });

  const K_SCALE_SIZE = 12;
  const TENSOR_BLOCKS = 256;
  const QK_K = 256;

  const ds_offset = 0;
  const ds_length = 4 * TENSOR_BLOCKS;
  const dmins_offset = ds_length;
  const dmins_length = 4 * TENSOR_BLOCKS;
  const scales_offset = dmins_offset + dmins_length;
  const scales_length = TENSOR_BLOCKS * K_SCALE_SIZE;
  const qs_offset = scales_offset + scales_length;
  const qs_length = (TENSOR_BLOCKS * QK_K) / 2;
  const offsetsAndLengths = [
    [ds_offset, ds_length],
    [dmins_offset, dmins_length],
    [scales_offset, scales_length],
    [qs_offset, qs_length],
  ];
  console.log(JSON.stringify(offsetsAndLengths));

  // create a buffer on the GPU to hold our computation
  // input and output
  const buffers = offsetsAndLengths.map(([offset, length], i) => {
    const buffer = device.createBuffer({
      label: `input-buffer-${i}`,
      size: length,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    // Copy our input data to that buffer
    device.queue.writeBuffer(buffer, 0, input, offset, length);
    return buffer;
  });

  // create a buffer on the GPU to get a copy of the results

  const dequantised = new Float32Array(256 * 256);
  const dequantisedBuffer = device.createBuffer({
    label: "dequantised buffer",
    size: dequantised.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // create a buffer on the GPU to get a copy of the results
  const resultBuffer = device.createBuffer({
    label: "result buffer",
    size: dequantised.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const debug = new Uint32Array(8);
  const debugBuffer = device.createBuffer({
    label: "debug buffer",
    size: debug.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // create a buffer on the GPU to get a copy of the results
  const debugResultBuffer = device.createBuffer({
    label: "debug result buffer",
    size: debug.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    label: "bindGroup for work buffer",
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: buffers[0] } },
      { binding: 1, resource: { buffer: buffers[1] } },
      { binding: 2, resource: { buffer: buffers[2] } },
      { binding: 3, resource: { buffer: buffers[3] } },
      {
        binding: 4,
        resource: { buffer: dequantisedBuffer },
      },
      {
        binding: 5,
        resource: { buffer: debugBuffer },
      },
    ],
  });

  // Encode commands to do the computation
  const encoder = device.createCommandEncoder({
    label: "dequant_q4k encoder",
  });
  const pass = encoder.beginComputePass({
    label: "dequant_q4k compute pass",
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(input.length);
  pass.end();

  encoder.copyBufferToBuffer(
    dequantisedBuffer,
    0,
    resultBuffer,
    0,
    resultBuffer.size
  );

  encoder.copyBufferToBuffer(
    debugBuffer,
    0,
    debugResultBuffer,
    0,
    debugResultBuffer.size
  );
  // Finish encoding and submit the commands
  const commandBuffer = encoder.finish();
  device.queue.submit([commandBuffer]);

  // Read the results
  await resultBuffer.mapAsync(GPUMapMode.READ);
  await debugResultBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(resultBuffer.getMappedRange());
  const debugResult = new Uint32Array(debugResultBuffer.getMappedRange());

  console.log("input", input);
  console.log("result", result);
  console.log("debugResult", debugResult);

  resultBuffer.unmap();
}

main();
