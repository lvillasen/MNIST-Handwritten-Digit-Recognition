import { MnistData } from "./data.js";

const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
];
let barChart;
const progressBar = document.getElementById("progressBar");

var toggle_code1 = document.getElementById("toggleCode1");
toggle_code1.addEventListener("click", code_out1);
var load_bt = document.getElementById("load");
load_bt.addEventListener("click", loadData);

var toggle_code2 = document.getElementById("toggleCode2");
toggle_code2.addEventListener("click", code_out2);
var get_model_bt = document.getElementById("model");
get_model_bt.addEventListener("click", get_model);

var toggle_code3 = document.getElementById("toggleCode3");
toggle_code3.addEventListener("click", code_out3);
var train_model_bt = document.getElementById("train");
train_model_bt.addEventListener("click", train_model);

var toggle_code4 = document.getElementById("toggleCode4");
toggle_code4.addEventListener("click", code_out4);
var evaluate_model_bt = document.getElementById("evaluate");
evaluate_model_bt.addEventListener("click", evaluate_model);

var toggle_code5 = document.getElementById("toggleCode5");
toggle_code5.addEventListener("click", code_out5);
var classify_bt = document.getElementById("classify");
classify_bt.addEventListener("click", classify);

var clear_bt = document.getElementById("clearButton");
clear_bt.addEventListener("click", clear);

var open_bt = document.getElementById("open_visor");
open_bt.addEventListener("click", open_visor);

var close_bt = document.getElementById("close_visor");
close_bt.addEventListener("click", close_visor);

document.getElementById("code1").style.display = "none";
document.getElementById("code2").style.display = "none";
document.getElementById("code3").style.display = "none";
document.getElementById("code4").style.display = "none";
document.getElementById("code5").style.display = "none";

var data;
var model;
async function showExamples(data) {
  const n_images = parseInt(document.getElementById("n_images").value);
  const surface = tfvis
    .visor()
    .surface({ name: "Input Data Examples", tab: "Input Data" });

  const examples = data.nextTestBatch(n_images);
  const numExamples = examples.xs.shape[0];

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    if (i == 1) {
    }

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);
}

function getModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(tf.layers.flatten());

  const NUM_OUTPUT_CLASSES = 10;
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );

  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function train(model, data) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = {
    name: "Training",
    tab: "Model",
    styles: { height: "1000px" },
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = parseInt(document.getElementById("n_batch").value);
  //const TRAIN_DATA_SIZE = 6000;
  const TRAIN_DATA_SIZE = parseInt(document.getElementById("n_train").value);
  const TEST_DATA_SIZE = parseInt(document.getElementById("n_test").value);
  const totalEpochs = parseInt(document.getElementById("n_epochs").value);
  var n_batch = 0;
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: totalEpochs,
    shuffle: true,
    callbacks: [
      fitCallbacks,
      {
        onBatchEnd: async (batch, logs) => {
          n_batch++;
          var progress =
            (n_batch / ((TRAIN_DATA_SIZE * totalEpochs) / BATCH_SIZE)) * 100;
          progressBar.value = progress; // Actualizar la barra de progreso
        },
      },
    ],
  });
}

function doPrediction(model, data, testDataSize = 2000) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1,
  ]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = { name: "Accuracy", tab: "Evaluation" };
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = { name: "Confusion Matrix", tab: "Evaluation" };
  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: classNames,
  });

  labels.dispose();
}

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

const canvasSize = Math.min(window.innerWidth, window.innerHeight) * 0.4;
canvas.width = canvasSize;
canvas.height = canvasSize;

let isDrawing = false;
ctx.lineWidth = 15; // Grosor del trazo
ctx.lineCap = "round"; // Línea redondeada
ctx.strokeStyle = "white"; // Color del trazo

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

function startDrawing(e) {
  isDrawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
}

function draw(e) {
  if (!isDrawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
}

function stopDrawing() {
  isDrawing = false;
  ctx.closePath();
}

bar_chart([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
bar_chart([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);

function code_out1() {
  var codeOut = document.getElementById("code1");
  if (codeOut.style.display === "none") {
    codeOut.style.display = "block";
  } else {
    codeOut.style.display = "none";
  }
}

function code_out2() {
  var codeOut = document.getElementById("code2");
  if (codeOut.style.display === "none") {
    codeOut.style.display = "block";
  } else {
    codeOut.style.display = "none";
  }
}

function code_out3() {
  var codeOut = document.getElementById("code3");
  if (codeOut.style.display === "none") {
    codeOut.style.display = "block";
  } else {
    codeOut.style.display = "none";
  }
}

function code_out4() {
  var codeOut = document.getElementById("code4");
  if (codeOut.style.display === "none") {
    codeOut.style.display = "block";
  } else {
    codeOut.style.display = "none";
  }
}

function code_out5() {
  var codeOut = document.getElementById("code5");
  if (codeOut.style.display === "none") {
    codeOut.style.display = "block";
  } else {
    codeOut.style.display = "none";
  }
}

async function loadData() {
  data = new MnistData();
  await data.load();
  await showExamples(data);
}

async function get_model() {
  model = getModel();
  tfvis.show.modelSummary({ name: "Model Architecture", tab: "Model" }, model);
}

async function train_model() {
  //tfvis.show.modelSummary({name: 'Model Training', tab: 'Model Training'}, model);
  await train(model, data);
}

async function evaluate_model() {
  await showAccuracy(model, data);
  await showConfusion(model, data);
}

function clear() {
  const canvas = document.getElementById("drawingCanvas");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
function open_visor() {
  tfvis.visor().open();
}
function close_visor() {
  tfvis.visor().close();
}
clear();

async function classify() {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  tempCanvas.style = "margin: 4px;";
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, 0, 0, 28, 28);
  tempCtx.fillStyle = "black";
  const outputCanvas = document.getElementById("outputCanvas");
  const outputCtx = outputCanvas.getContext("2d");
  outputCtx.fillStyle = "black";
  outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
  outputCtx.drawImage(tempCanvas, 0, 0);
  const surface = tfvis.visor().surface({ name: "Test Images", tab: "Test" });
  const canvas1 = document.createElement("canvas");
  canvas1.width = 28;
  canvas1.height = 28;
  canvas1.style = "margin: 4px;";
  surface.drawArea.appendChild(tempCanvas);
  clear();
  predictCanvasImage();
}

async function predictCanvasImage() {
  const outputCanvas = document.getElementById("outputCanvas");
  let tensor = tf.browser.fromPixels(outputCanvas, 1);
  tensor = tensor.toFloat().div(tf.scalar(255));
  tensor = tensor.expandDims(0);
  try {
    const prediction = model.predict(tensor);
    const predictedDigit = prediction.argMax(1).dataSync()[0];
    document.getElementById("prediction").innerText = predictedDigit;
    bar_chart(prediction.dataSync());
  } catch (err) {
    bar_chart([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
  }
}
function bar_chart(Values) {
  if (barChart) {
    barChart.destroy();
  }
  const rainbowColors = [
    "rgba(255, 0, 0, 0.7)", // Rojo
    "rgba(255, 127, 0, 0.7)", // Naranja
    "rgba(255, 255, 0, 0.7)", // Amarillo
    "rgba(0, 255, 0, 0.7)", // Verde
    "rgba(0, 255, 255, 0.7)", // Cian
    "rgba(0, 127, 255, 0.7)", // Azul claro
    "rgba(0, 0, 255, 0.7)", // Azul
    "rgba(127, 0, 255, 0.7)", // Violeta claro
    "rgba(148, 0, 211, 0.7)", // Índigo
    "rgba(75, 0, 130, 0.7)", // Violeta oscuro
  ];
  const labels = [...Array(10).keys()];
  const dataBar = {
    labels: labels,
    datasets: [
      {
        label: "Prediction ",
        data: Values,
        backgroundColor: rainbowColors,
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
      },
    ],
  };
  const options = {
    scales: {
      y: {
        beginAtZero: true, // Comenzar el eje Y en 0
        max: 1, // Limitar el eje Y al valor máximo 1
      },
    },
  };
  const ctxBar = document.getElementById("barChart").getContext("2d");
  barChart = new Chart(ctxBar, {
    type: "bar", // Tipo de gráfico: barra
    data: dataBar, // Datos del gráfico
    options: options, // Opciones del gráfico
  });
}
