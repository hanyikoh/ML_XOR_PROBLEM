let inputs;
const resolution = 25;
let cols;
let rows;

const train_xs = tf.tensor2d([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

const train_ys = tf.tensor2d([
    [0],
    [1],
    [1],
    [0]
])

let xs;

async function setup() {
    createCanvas(400, 400);
    inputs = []
    cols = width / resolution
    rows = height / resolution
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let x1 = i / cols;
            let x2 = j / rows;
            inputs.push([x1, x2])
        }
    }
    ///stroke(0);
    xs = tf.tensor2d(inputs);

    model = tf.sequential();
    let hidden = tf.layers.dense({
        inputShape: [2],
        units: 4,
        activation: 'sigmoid'
    });
    let output = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    });
    model.add(hidden);
    model.add(output);

    const optimizer = tf.train.adam(0.1);

    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError'
    })

    setTimeout(train, 100)
}

function train(){
    trainModel().then(result =>{
        console.log(result.history.loss[0])
        setTimeout(train, 100)
    })}

function trainModel() {
    return model.fit(train_xs, train_ys, {
        shuffle: true,
        epochs: 100
    })
}

function draw() {
    background(0)
    tf.tidy(() => {
        trainModel().then(h => console.log(h.history.loss[0]));
    })
    //noLoop()
    tf.tidy(() => {
        let ys = model.predict(xs)
        let ys_values = ys.dataSync()

        let index = 0;
        for (let i = 0; i < cols; i++) {
            for (let j = 0; j < rows; j++) {
                let br = ys_values[index] * 255
                fill(br);
                rect(i * resolution, j * resolution, resolution, resolution)
                fill(255-br);
                textSize(8)
                textAlign(CENTER, CENTER)
                text(nf(ys_values[index], 1,2), i * resolution + resolution/2, j * resolution + resolution/2 )
                index++
            }
        }
    })
    //noLoop()
}

