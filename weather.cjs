const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-wasm');

const fs = require('fs');
const axios = require('axios');
require('dotenv').config();

const API_KEY = '6D7KM323XCEH5ANH7F2WCVAZN';
const location = 'New York City,NY'; // change this
const MAX_TEMP = 50; // for normalization

let history = [];

async function fetchWeatherHistory() {

    const endDate = new Date().toISOString().split('T')[0];
    const startDate = new Date(Date.now() - 31 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
    const elements = "temp,pressure,humidity,datetime";
    const url = `https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/${location}/${startDate}/${endDate}?unitGroup=metric&key=${API_KEY}&contentType=json&include=days&elements=${elements}`;

    try {
        console.log('Fetching weather history...');
        const response = await axios.get(url);

        const processedData = response.data.days.map(day => {
            const pressure = day.pressure || (response.data.currentConditions && response.data.currentConditions.pressure) || 1013.2;
            
            return {
                temp: day.temp,
                press: pressure,
                humid: day.humidity || 50,
                date: day.datetime
            };
        });

        fs.writeFileSync('weather_history.json', JSON.stringify(processedData, null, 2));
        console.log('Saved weather history with pressure data!');
    } catch (error) {
        console.error('Error:', error.message);
    }
}


//reinforcement training thing

//save and load data

async function saveModel(model) {
    const modelPath = './models/weather-model';
    if (!fs.existsSync(modelPath)) {
        fs.mkdirSync(modelPath, { recursive: true });
    }
    
    const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
        return { modelArtifacts: artifacts };
    }));

    const modelData = {
        topology: saveResult.modelArtifacts.modelTopology,
        weights: Buffer.from(saveResult.modelArtifacts.weightData).toString('base64')
    };

    fs.writeFileSync(`${modelPath}/brain.json`, JSON.stringify(modelData));
    console.log("Model brain saved to disk manually.");
}

async function loadModel() {
    const filePath = './models/weather-model/brain.json';
    if (!fs.existsSync(filePath)) {
        console.log("No saved brain found.");
        return null;
    }
    const modelData = JSON.parse(fs.readFileSync(filePath));
    
    const weightData = new Uint8Array(Buffer.from(modelData.weights, 'base64')).buffer;

    const model = await tf.loadLayersModel(tf.io.fromMemory({
        modelTopology: modelData.topology,
        weightSpecs: modelData.topology.weightsManifest[0].weights, 
        weightData: weightData
    }));

    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    console.log("Model brain loaded manually.");
    return model;
}

async function loadData() {
    if (fs.existsSync('weather_history.json')) {
        const rawData = fs.readFileSync('weather_history.json');
        history = JSON.parse(rawData);
        console.log("Weather history loaded.");
    } else {
        console.log("There exists no weather JSON");
    }
}

async function pickOptimalGPU() {
    if (navigator.gpu) {
        await tf.setBackend('webgpu');
    } else {
        await tf.setBackend('webgl');
    }
    console.log("Using GPU:", tf.getBackend());
}
pickOptimalGPU();

//old
// async function trainModel() {
//     const model = tf.sequential();
//     model.add(tf.layers.dense({ units: 12, inputShape: [3], activation: 'relu' }));
//     model.add(tf.layers.dense({ units: 1 }));
//     model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

//     console.log("Training model...");

//     for (let i = 0; i < history.length-1; i++) {
//         const today = history[i];
//         const actualTomorrow = history[i+1].temp;

//         //predict
//         const input = tf.tensor2d([[today.temp/MAX_TEMP, today.press/1100, today.humid/100]]);
//         const predictionTensor = model.predict(input);
//         const predictedTemp = (predictionTensor.dataSync()[0] * MAX_TEMP).toFixed(1);

//         const target = tf.tensor2d([[actualTomorrow / MAX_TEMP]]);

//         await model.fit(input, target, { epochs: 5, verbose: 0 });

//         console.log(`Day ${i}: Predicted ${predictedTemp}°C | Actual ${actualTomorrow}°C | Error: ${Math.abs(predictedTemp - actualTomorrow).toFixed(1)}`);
//     }

//     console.log('Finished training');

//     saveModel(model);
// }

async function trainModel(history) {
    const MAX_TEMP = 50; // Max expected temperature
    const MAX_PRESS = 1100; // Max expected pressure
    const MAX_HUMID = 100; // Max expected humidity

    // 1. Setup the Model Architecture
    const model = tf.sequential();
    
    // Hidden Layer 1: 16 neurons to find initial patterns
    model.add(tf.layers.dense({ 
        units: 16, 
        inputShape: [3], 
        activation: 'relu' 
    }));
    
    // Hidden Layer 2: 8 neurons to refine the logic
    model.add(tf.layers.dense({ 
        units: 8, 
        activation: 'relu' 
    }));
    
    // Output Layer: 1 neuron for the predicted temperature
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ 
        optimizer: tf.train.adam(0.005), 
        loss: 'meanSquaredError' 
    });

    // 2. Prepare and Normalize Data from JSON
    const inputs = [];
    const labels = [];

    for (let i = 0; i < history.length - 1; i++) {
        const today = history[i];
        const tomorrow = history[i + 1];

        // Normalize all values between 0 and 1
        inputs.push([
            today.temp / MAX_TEMP, 
            today.press / MAX_PRESS, 
            today.humid / MAX_HUMID
        ]);

        // The target is tomorrow's temperature (also normalized)
        labels.push([tomorrow.temp / MAX_TEMP]);
    }

    // Convert arrays to Tensors
    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(labels);

    console.log(`Training on ${history.length} days of data...`);

    await model.fit(xs, ys, {
        epochs: 100,    
        shuffle: true,    
        batchSize: 32,  
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (epoch % 10 === 0) {
                    console.log(`Epoch ${epoch}: Loss = ${logs.loss.toFixed(6)}`);
                }
            }
        }
    });

    console.log('Training complete!');
    xs.dispose();
    ys.dispose();
    if (typeof saveModel === 'function') {
        saveModel(model);
    }
    
    return model;
}


async function main() {
    console.log("Initializing WASM...");
    await tf.setBackend('wasm');
    await tf.ready(); 
    console.log("Backend ready: ", tf.getBackend());

    // Make sure we have data
    if (!fs.existsSync('weather_history.json')) {
        await fetchWeatherHistory();
    }
    
    await loadData();
    let model = await loadModel();
    if (!model) {
        await trainModel(); 
    } else {
        console.log("Brain already exists. Ready for predictions or further training.");
        await trainModel(); 
    }
}

main();
