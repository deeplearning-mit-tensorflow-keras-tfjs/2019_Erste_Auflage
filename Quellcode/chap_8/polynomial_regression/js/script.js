/** 
 * 
 *  Lineare regression 
 */

 
let data;

const numbers = 10;

function predictAndDraw(input, model) {

  var resArray = [];
  for (var p = 0; p < numbers; p++)
    resArray.push(model.predict(tf.tensor2d([p], [1, 1])).dataSync());

  data.labels.push(Array.from(input.dataSync()))
  data.series.push({
    className: "ct-series-b",
    data: Array.from(resArray)
  })

  chart.update(data);
  data.labels.pop();
  data.series.pop();
}

/** Generierung eines randomisierte Datensets */
(async () => {

  var x = tf.range(0, numbers, 1).reshape([numbers, 1]);
  const y = x.pow(tf.tensor(2)).add(tf.tensor(5)); //.pow(tf.tensor(2));
  //y.print()
  // Kurve wird dargestellt 
  drawCurve(x, y)

  // Aufbau des Modells
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 10,
    inputShape: [1],
  }));

  model.add(tf.layers.activation({
    activation: "sigmoid"
  }));
  /*
  model.add(tf.layers.dense({
    units: 50,
    inputShape: [1],
  }));

  model.add(tf.layers.dropout({
    rate: 0.5
  }));
    
  model.add(tf.layers.activation({
    activation: "tanh"
  }));
  */
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1],
  }));


  const learningRate = 0.0025
  const opti = tf.train.sgd(learningRate);

  // Vorbereitung des Modells für das Training
  model.compile({
    loss: 'meanSquaredError',
    optimizer: opti,
    metrics: ['accuracy']
  });

  // Ausgabe der Modell-Struktur
  model.summary();

  // Training des Modells
  await model.fit(x, y, {
    epochs: 10000,
    batchSize: 5,
    validationData: [x, y],
    callbacks: {

      // Am Ende jedes Epochs wird die Anzeige aktualisiert
      onEpochEnd: async (epoch, logs) => {

        $("#epochs").html("<span>Epochs: " + epoch + "</span>");
        $("#loss").html("<span>Loss: " + logs.loss + "</span>");

        predictAndDraw(x, model);
        var r = model.predict(tf.tensor2d([5], [1, 1]));
        // r.print();
        await tf.nextFrame();
      }

    }
  });
  // Optional: Speichern des Modells am Ende des Trainings
  // await model.save("downloads://my-model")
})();