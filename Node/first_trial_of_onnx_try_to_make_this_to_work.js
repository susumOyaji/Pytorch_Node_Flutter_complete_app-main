// Reference for this file
// https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
import * as Jimp from 'jimp';
// Define "require"
import { createRequire } from "module";
const require = createRequire(import.meta.url);

require("onnxjs");
require("onnxjs-node");

const getImageTensorFromPath = async (path, dims =  [1, 3, 224, 224])=>{
    var image = await loadImageFromPath(path, dims[2], dims[3]);
    var imageTensor = imageDataToTensor(image, dims);
    return imageTensor;
  }
  
const loadImageFromPath = async (path, width = 224, height = 224)=>{
    // Use Jimp to load the image and resize it.
    var imageData = await Jimp.default.read(path).then((imageBuffer) => {
      return imageBuffer.resize(width, height);
    });
    return imageData;
  }
  
const imageDataToTensor = async(image, dims)=>{
    var imageBufferData = image.bitmap.data;
    let [redArray, greenArray, blueArray] = new Array(
        new Array(), 
        new Array(), 
        new Array());
    // 2. Loop through the image buffer and extract the R, G, and B channels
    for (let i = 0; i < imageBufferData.length; i += 4) {
      redArray.push(imageBufferData[i]);
      greenArray.push(imageBufferData[i + 1]);
      blueArray.push(imageBufferData[i + 2]);
      // skip data[i + 3] to filter out the alpha channel
    }
  
    // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
    const transposedData = redArray.concat(greenArray).concat(blueArray);
  
    // 4. convert to float32
    let l = transposedData.length; // length, we need this for the loop
    // create the Float32Array size 3 * 224 * 224 for these dimensions output
    const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
    for (let i = 0; i < l; i++) {
      float32Data[i] = transposedData[i] / 255.0; // convert to float
    }
    // 5. create the tensor object from onnxruntime-web.
    const inputTensor = new onnx.Tensor("float32", float32Data, dims);
    return inputTensor;
}

export { getImageTensorFromPath };