import express from "express";
// Define "require"
import { createRequire } from "module";
const require = createRequire(import.meta.url);

const ort = require('onnxruntime-node'),
    Jimp = require("jimp"),
    model_path = '../model.onnx',
    fs = require('fs'),
    port = process.env.PORT || 8085,
    bodyParser = require("body-parser"),
    fileUpload = require("express-fileupload"),
    app = express();

app.use(bodyParser.urlencoded({ extended: true, limit: "50mb" }));
app.use(fileUpload());

let main = async () => {
    return ort.InferenceSession.create(model_path)
        .then((data) => {
            return data;
        }, error => { console.log("Model error! Check if file exists") });
}


async function predictImage(image){
    const session = await main();
    let imageBufferData = image.data;
    // 1. Get buffer data from image and create R, G, and B arrays.
    let flattenedData = new Array();
    // 2. Loop through the image buffer and extract the R, G, and B channels
    for (let i = 0; i < imageBufferData.length; i += 4) {
        flattenedData.push(imageBufferData[i]);
        flattenedData.push(imageBufferData[i + 1]);
        flattenedData.push(imageBufferData[i + 2]);
        // skip data[i + 3] to filter out the alpha channel
    }
    const dataA = Float32Array.from(flattenedData);
    const tensorA = new ort.Tensor('float32', dataA, [224, 224, 3]);
    // print model to get input and output
    const feeds = { 'img.1': tensorA };
    const results = await session.run(feeds);
    let predicted_output = 1 / (1 + Math.exp(-(results['486'].data[0])));
    // unary + operator, which converts its operand into a number.
    predicted_output = ["Cat", "Dog"][+ (predicted_output >= 0.5)]
    return predicted_output;
}

let ImageResizer = async (data) => {
    try {
      return await Jimp.read(data).then((data) => {
        return data.resize(224, 224).bitmap;
      });
      
    } catch (e) {
      console.log(e);
    }
  };

app.post("/predict", async (req, res) => {
    let img = req.body.image;
    let file = Buffer.from(img,"base64");
    file = await ImageResizer(file);
    let prediction = await predictImage(file);
    console.log(prediction);
    res.send(prediction);
});

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`);
});