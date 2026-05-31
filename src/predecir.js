const tf   = require('@tensorflow/tfjs-node');
const path = require('path');
const fs   = require('fs');
const Jimp = require('jimp');

const SAVE_DIR = path.join(__dirname, '..', 'modelo_guardado');

let modeloCargado = null;
let meta          = null;

async function cargarModelo() {
  if (modeloCargado) return;

  const metaPath = path.join(SAVE_DIR, 'meta.json');
  if (!fs.existsSync(metaPath)) {
    throw new Error('Modelo no encontrado. Ejecuta: npm run train');
  }

  meta          = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  modeloCargado = await tf.loadLayersModel(
    `file://${path.join(SAVE_DIR, 'model.json')}`
  );

  // Detectar tamaño real del modelo automáticamente
  const inputShape = modeloCargado.inputs[0].shape;
  const imgSizeReal = inputShape[1]; // [null, H, W, C] → H
  meta.imgSize = imgSizeReal;

  console.log(`✅ Modelo cargado | inputShape: ${JSON.stringify(inputShape)} | imgSize detectado: ${imgSizeReal}px`);
}

async function preprocesar(buffer) {
  // Usar el tamaño REAL del modelo, no el del meta.json
  const imgSize = meta?.imgSize || 128;

  const img = await Jimp.read(buffer);
  img.resize(imgSize, imgSize).grayscale();

  const pixeles = new Float32Array(imgSize * imgSize);
  let i = 0;
  img.scan(0, 0, img.bitmap.width, img.bitmap.height, (x, y, idx) => {
    pixeles[i++] = img.bitmap.data[idx] / 255.0;
  });

  return tf.tensor4d(pixeles, [1, imgSize, imgSize, 1]);
}

async function predecir(buffer) {
  await cargarModelo();

  const tensor     = await preprocesar(buffer);
  const prediccion = modeloCargado.predict(tensor);
  const probCancer = (await prediccion.data())[0];

  tensor.dispose();
  prediccion.dispose();

  const threshold = 0.5;
  const esCancer  = probCancer >= threshold;
  const clase     = esCancer ? 'cancer' : 'no_cancer';
  const confianza = parseFloat(
    (esCancer ? probCancer : 1 - probCancer) * 100
  ).toFixed(2);

  return {
    clase,
    confianza:      parseFloat(confianza),
    probabilidades: {
      cancer:    parseFloat((probCancer * 100).toFixed(2)),
      no_cancer: parseFloat(((1 - probCancer) * 100).toFixed(2))
    },
    valorSigmoid: parseFloat(probCancer.toFixed(4))
  };
}

module.exports = { predecir, cargarModelo };
