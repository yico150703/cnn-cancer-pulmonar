// src/modelo.js
// Arquitectura CNN basada en: kaggle.com/code/truenhatnam/cnn-lungcancer
// Clasificación binaria: cancer (1) / no_cancer (0)
// Entrada: imágenes en escala de grises 224×224×1

const tf = require('@tensorflow/tfjs');

/**
 * build_lung_mri_model — equivalente exacto del notebook Python:
 *
 *   def build_lung_mri_model(input_shape=(224,224,1), num_classes=1):
 *     model = Sequential()
 *     Conv2D(32) → BN → MaxPool
 *     Conv2D(64) → BN → MaxPool
 *     Conv2D(128) → BN → MaxPool → Dropout(0.3)
 *     GlobalAveragePooling2D
 *     Dense(256, relu) → Dropout(0.5)
 *     Dense(1, sigmoid)
 *
 * @param {number} imgSize - tamaño de imagen (default 224)
 * @returns {tf.Sequential}
 */
function buildLungMriModel(imgSize = 224) {
  const model = tf.sequential({ name: 'CNN_LungCancer_MRI' });

  // ── Bloque 1: Conv2D(32) + BN + MaxPool ──────────────────────────────────
  model.add(tf.layers.conv2d({
    filters:    16,
    kernelSize: [3, 3],
    activation: 'relu',
    padding:    'same',
    inputShape: [imgSize, imgSize, 1]   // 1 canal = escala de grises
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // ── Bloque 2: Conv2D(64) + BN + MaxPool ──────────────────────────────────
  model.add(tf.layers.conv2d({
    filters:    32,
    kernelSize: [3, 3],
    activation: 'relu',
    padding:    'same'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // ── Bloque 3: Conv2D(128) + BN + MaxPool + Dropout(0.3) ──────────────────
  model.add(tf.layers.conv2d({
    filters:    64,
    kernelSize: [3, 3],
    activation: 'relu',
    padding:    'same'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.3 }));

  // ── Clasificador ──────────────────────────────────────────────────────────
  model.add(tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.5 }));

  // ── Salida binaria (sigmoid) ───────────────────────────────────────────────
  // sigmoid → valor entre 0 y 1
  // < 0.5 → no_cancer | >= 0.5 → cancer
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  // ── Compilar con Adam(lr=1e-4) y binary_crossentropy ─────────────────────
  model.compile({
    optimizer: tf.train.adam(1e-4),
    loss:      'binaryCrossentropy',
    metrics:   ['accuracy']
  });

  return model;
}

module.exports = { buildLungMriModel };