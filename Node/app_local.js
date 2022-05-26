// Define "require"
import express from "express";
import { createRequire } from "module";
const require = createRequire(import.meta.url);

const ort = require('onnxruntime-node'),
    Jimp = require("jimp"),
    model_path = '../model.onnx',
    fs = require('fs'),
    port = process.env.PORT || 8080,
    app = express();

let main = async () => {
    return ort.InferenceSession.create(model_path)
        .then((data) => {
            return data;
        }, error => { console.log("Model error! Check if file exists") });
}

let paths;
const session = await main();

/* Randomize array in-place using Durstenfeld shuffle algorithm */
function shuffle(array) {
    let currentIndex = array.length, randomIndex;
    // While there remain elements to shuffle.
    while (currentIndex != 0) {
        // Pick a remaining element.
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        // And swap it with the current element.
        [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
    }
    return array;
}


try {
    paths = fs.readFileSync('test_paths.txt', 'utf8');
    paths = paths.split(/\r?\n/);
    shuffle(paths);
    paths = paths.slice(0, 16);
} catch (err) {
    console.error(err);
}

async function predictImage(path, image) {
    // 1. Get buffer data from image and create R, G, and B arrays.
    let imageBufferData = image.data;
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
    let correct_output = path.split("\\").slice(-2, -1);
    // unary + operator, which converts its operand into a number.
    predicted_output = ["Cat", "Dog"][+ (predicted_output >= 0.5)]
    if (predicted_output == correct_output)
        console.log(`${path} : \x1b[32m${predicted_output}\x1b[0m`);
    else
        console.log(`${path} : \x1b[31m${predicted_output}\x1b[0m`);
}

let inferencePipeline = async () => {
    for (let path of paths) {
        path = `../../../data/${path.slice(3)}`;
        let image = await Jimp.read(path)
            .then(data => {
                return data.resize(224, 224).bitmap;
            });
        await predictImage(path, image);
    }
};

await inferencePipeline();
