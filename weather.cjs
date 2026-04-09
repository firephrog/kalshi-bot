const tf   = require('@tensorflow/tfjs-node');
const fs   = require('fs');
const path = require('path');
const axios = require('axios');
require('dotenv').config();

// Config

const API_KEY  = process.env.WEATHER_API_KEY  || '6D7KM323XCEH5ANH7F2WCVAZN';
const LOCATION = process.env.WEATHER_LOCATION || 'New York City,NY';

const WINDOW_SIZE     = 14;
const FEATURES_PER_DAY = 11;
const FORECAST_DAYS   = 5;

const MODEL_PATH   = path.join(__dirname, 'models', 'weather-model');
const HISTORY_PATH = path.join(__dirname, 'weather_history.json');

const NORM = {
    temp:   { min: -30, max: 50   },
    press:  { min: 960, max: 1060 },
    humid:  { min: 0,   max: 100  },
    wind:   { min: 0,   max: 100  },
    cloud:  { min: 0,   max: 100  },
    precip: { min: 0,   max: 50   },
};

// Normalization

function normalize(val, n) {
    const clamped = Math.min(Math.max(val, n.min), n.max);
    return (clamped - n.min) / (n.max - n.min);
}

function denormalize(val, n) {
    return val * (n.max - n.min) + n.min;
}

// Featureization

function featurize(day) {
    const date     = new Date(day.date);
    const start    = new Date(date.getFullYear(), 0, 0);
    const dayOfYear = Math.floor((date - start) / 86400000);

    const yearAngle = (2 * Math.PI * dayOfYear) / 365;
    const weekAngle = (2 * Math.PI * date.getDay()) / 7;

    return [
        normalize(day.temp,      NORM.temp),
        normalize(day.press,     NORM.press),
        normalize(day.humid,     NORM.humid),
        normalize(day.wind,      NORM.wind),
        normalize(day.cloud,     NORM.cloud),
        normalize(day.precip,    NORM.precip),
        Math.min(Math.max(day.tempDelta / 20, -1), 1), // rate of change
        Math.sin(yearAngle),
        Math.cos(yearAngle),
        Math.sin(weekAngle),
        Math.cos(weekAngle),
    ];
}

// Grab data from API

async function fetchWeatherHistory(years = 3) {
    if (!API_KEY) throw new Error('WEATHER_API_KEY not set in .env');

    const allDays = [];
    const now = new Date();

    for (let y = 0; y < years; y++) {
        const end   = new Date(now);
        end.setFullYear(end.getFullYear() - y);
        const start = new Date(end);
        start.setFullYear(start.getFullYear() - 1);

        const endStr   = end.toISOString().split('T')[0];
        const startStr = start.toISOString().split('T')[0];

        const url = [
            `https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/`,
            `${encodeURIComponent(LOCATION)}/${startStr}/${endStr}`,
            `?unitGroup=metric&key=${API_KEY}&contentType=json&include=days`,
            `&elements=temp,pressure,humidity,datetime,windspeed,cloudcover,precip`,
        ].join('');

        console.log(`Fetching year ${y + 1}/${years}: ${startStr} → ${endStr}`);
        const response = await axios.get(url);

        const days = response.data.days.map(day => ({
            date:   day.datetime,
            temp:   day.temp       ?? 15,
            press:  day.pressure   ?? 1013,
            humid:  day.humidity   ?? 50,
            wind:   day.windspeed  ?? 0,
            cloud:  day.cloudcover ?? 50,
            precip: day.precip     ?? 0,
            tempDelta: 0, 
        }));

        allDays.push(...days);

        if (y < years - 1) await new Promise(r => setTimeout(r, 1200));
    }

    const unique = [...new Map(allDays.map(d => [d.date, d])).values()]
        .sort((a, b) => a.date.localeCompare(b.date));

    for (let i = 1; i < unique.length; i++) {
        unique[i].tempDelta = unique[i].temp - unique[i - 1].temp;
    }
    unique[0].tempDelta = 0;

    fs.writeFileSync(HISTORY_PATH, JSON.stringify(unique, null, 2));
    console.log(`Saved ${unique.length} days of history.\n`);
    return unique;
}

// Data set

function buildDataset(history) {
    const inputs = [];
    const labels = [];

    for (let i = WINDOW_SIZE; i < history.length; i++) {
        const window = history.slice(i - WINDOW_SIZE, i).map(day => featurize(day));
        inputs.push(window);
        labels.push([
            normalize(history[i].temp,  NORM.temp),
            normalize(history[i].humid, NORM.humid),
            normalize(history[i].press, NORM.press),
        ]);
    }

    return {
        xs: tf.tensor3d(inputs),           // [n, WINDOW_SIZE, FEATURES_PER_DAY]
        ys: tf.tensor2d(labels),           // [n, 3]
    };
}

// AI Model

function buildModel() {
    const model = tf.sequential();

    // First LSTM
    model.add(tf.layers.lstm({
        units: 128,
        inputShape: [WINDOW_SIZE, FEATURES_PER_DAY],
        returnSequences: true,
        recurrentDropout: 0.1,
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));

    // Second LSTM
    model.add(tf.layers.lstm({
        units: 64,
        returnSequences: false,
        recurrentDropout: 0.1,
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    // Dense head
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

    // Output
    model.add(tf.layers.dense({ units: 3 }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
        metrics: ['mae'],
    });

    return model;
}

// Trainer

async function trainModel(history) {
    const { xs, ys } = buildDataset(history);
    const model = buildModel();

    console.log('Model summary:');
    model.summary();
    console.log(`\nTraining on ${xs.shape[0]} samples...\n`);

    let bestValLoss = Infinity;
    let epochsNoImprove = 0;
    const PATIENCE = 25;
    let bestWeights = null;

    await model.fit(xs, ys, {
        epochs: 300,
        batchSize: 32,
        validationSplit: 0.15,
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Log every 20 epochs
                if (epoch % 20 === 0) {
                    const maeC = (denormalize(logs.mae, NORM.temp) - NORM.temp.min).toFixed(2);
                    console.log(
                        `Epoch ${String(epoch).padStart(3)}: ` +
                        `loss=${logs.loss.toFixed(5)}  ` +
                        `val_loss=${(logs.val_loss ?? 0).toFixed(5)}  ` +
                        `MAE≈${maeC}°C`
                    );
                }

                // LR decay every 75 epochs
                if (epoch > 0 && epoch % 75 === 0) {
                    const oldLR = model.optimizer.learningRate;
                    model.optimizer.learningRate = oldLR * 0.5;
                    console.log(`  → LR decayed to ${model.optimizer.learningRate.toFixed(6)}`);
                }

                // Manual early stopping with best-weight restoration
                if (logs.val_loss !== undefined) {
                    if (logs.val_loss < bestValLoss) {
                        bestValLoss = logs.val_loss;
                        epochsNoImprove = 0;
                        if (bestWeights) bestWeights.forEach(w => w.dispose());
                        bestWeights = model.getWeights().map(w => w.clone());
                    } else {
                        epochsNoImprove++;
                        if (epochsNoImprove >= PATIENCE) {
                            console.log(`\nEarly stop at epoch ${epoch} — best val_loss=${bestValLoss.toFixed(5)}`);
                            model.stopTraining = true;
                        }
                    }
                }
            },
        },
    });

    // Restore best checkpoint
    if (bestWeights) {
        model.setWeights(bestWeights);
        bestWeights.forEach(w => w.dispose());
        console.log(`\nRestored best weights (val_loss=${bestValLoss.toFixed(5)})`);
    }

    xs.dispose();
    ys.dispose();

    if (!fs.existsSync(MODEL_PATH)) fs.mkdirSync(MODEL_PATH, { recursive: true });
    await model.save(`file://${MODEL_PATH}`);
    console.log(`Model saved to ${MODEL_PATH}\n`);

    return model;
}

// Eval

async function evaluate(model, history) {
    const TEST_DAYS = 14;
    const testHistory = history.slice(-(TEST_DAYS + WINDOW_SIZE));
    const { xs, ys } = buildDataset(testHistory);

    console.log(`── Evaluation on last ${TEST_DAYS} days ──`);
    let totalTempErr = 0;

    for (let i = 0; i < TEST_DAYS; i++) {
        const input = xs.slice([i, 0, 0], [1, WINDOW_SIZE, FEATURES_PER_DAY]);
        const pred  = model.predict(input);
        const [tNorm, hNorm, pNorm] = pred.dataSync();
        input.dispose();
        pred.dispose();

        const predTemp  = denormalize(tNorm, NORM.temp);
        const actualTemp = denormalize(ys.dataSync()[i * 3], NORM.temp);
        const err = Math.abs(predTemp - actualTemp);
        totalTempErr += err;

        const day = history[history.length - TEST_DAYS + i];
        console.log(
            `  ${day.date}: predicted=${predTemp.toFixed(1)}°C  ` +
            `actual=${actualTemp.toFixed(1)}°C  error=${err.toFixed(1)}°C`
        );
    }

    console.log(`  Mean Absolute Error: ${(totalTempErr / TEST_DAYS).toFixed(2)}°C\n`);
    xs.dispose();
    ys.dispose();
}

// Forecast

async function forecast(model, history) {
    const extended = [...history];

    console.log(`── ${FORECAST_DAYS}-Day Forecast for ${LOCATION} ──`);

    for (let d = 0; d < FORECAST_DAYS; d++) {
        const window = [extended.slice(-WINDOW_SIZE).map(day => featurize(day))];
        const input  = tf.tensor3d(window);
        const pred   = model.predict(input);
        const [tNorm, hNorm, pNorm] = pred.dataSync();
        input.dispose();
        pred.dispose();

        const temp  = denormalize(tNorm, NORM.temp);
        const humid = denormalize(hNorm, NORM.humid);
        const press = denormalize(pNorm, NORM.press);

        const lastDay  = extended[extended.length - 1];
        const nextDate = new Date(lastDay.date);
        nextDate.setDate(nextDate.getDate() + 1);

        extended.push({
            date:      nextDate.toISOString().split('T')[0],
            temp,
            press,
            humid,
            wind:      lastDay.wind,
            cloud:     lastDay.cloud,
            precip:    lastDay.precip,
            tempDelta: temp - lastDay.temp,
        });

        console.log(
            `  Day +${d + 1} (${nextDate.toISOString().split('T')[0]}): ` +
            `${temp.toFixed(1)}°C  ` +
            `${humid.toFixed(0)}% humidity  ` +
            `${press.toFixed(0)} hPa`
        );
    }
}

// Main

async function main() {
    console.log(`Backend: ${tf.getBackend()}\n`);

    const forceRetrain = process.argv.includes('--retrain');
    const forceFetch   = process.argv.includes('--fetch');

    // Load or fetch history
    let history;
    const historyExists = fs.existsSync(HISTORY_PATH);
    const historyAge    = historyExists
        ? Date.now() - fs.statSync(HISTORY_PATH).mtimeMs
        : Infinity;
    const historyStale  = historyAge > 24 * 60 * 60 * 1000;

    if (!historyExists || historyStale || forceFetch) {
        history = await fetchWeatherHistory(3);
    } else {
        history = JSON.parse(fs.readFileSync(HISTORY_PATH));
        console.log(`Loaded ${history.length} days from cache.\n`);
    }


    console.log('Checking for NaN/bad values in history...');
    history.forEach((day, i) => {
        const features = featurize(day);
        const hasNaN = features.some(f => isNaN(f) || !isFinite(f));
        if (hasNaN) {
            console.log(`Bad data at index ${i} (${day.date}):`, day, features);
        }
    });
    console.log('Check complete.');

    // Load or train model
    const modelJsonPath = path.join(MODEL_PATH, 'model.json');
    let model;

    if (fs.existsSync(modelJsonPath) && !forceRetrain && !forceFetch) {
        model = await tf.loadLayersModel(`file://${modelJsonPath}`);
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mae'],
        });
        console.log('Model loaded from disk.\n');
    } else {
        model = await trainModel(history);
    }

    await evaluate(model, history);
    await forecast(model, history);

    model.dispose();
}

main().catch(console.error);
