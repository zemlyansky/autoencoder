const Autoencoder = require('./index')
const fs = require('fs')
const tape = require('tape')

// Example dataset loaded from TH.data package
// https://cran.r-project.org/web/packages/TH.data/index.html
const X = fs
  .readFileSync('./data/birds.csv')
  .toString()
  .split('\n')
  .filter((line, i) => i > 0 && line.includes(','))
  .map(row => row
    .split(',')
    .splice(1)
    .map(x => parseFloat(x))
  )

tape('Test scaler', t => {
  const ae = new Autoencoder({
    'nInputs': X[0].length,
    'nHidden': 2
  })
  const Xn = [[1, -10, 8], [4, 9, 8], [6, 7, 8]]
  const Xt = ae._scalerFitTransform(Xn)
  const Xr = ae._scalerInverseTransform(Xt)
  t.plan(1)
  t.deepEqual(Xn, Xr)
  console.log(Xn, Xr)
})

tape('Autoencoder test (auto layers size)', t => {
  const ae = new Autoencoder({
    'nInputs': X[0].length,
    'nHidden': 2,
    'activation': 'tanh'
  })

  ae.fit(X, {
    'batchSize': 100,
    'iterations': 2000,
    'stepSize': 0.05
  })

  t.plan(3)

  const Y1 = ae.encode(X)
  const Y2 = ae.decode(Y1)
  const Y3 = ae.predict(X)

  t.equal(Y1[0].length, 2)
  t.equal(Y2[0].length, X[0].length)
  t.equal(Y3[0].length, X[0].length)

  console.log(X[0], Y1[0], Y2[0], Y3[0])
})

tape('Autoencoder test (manual layers design)', t => {
  const ae = new Autoencoder({
    'encoder': [
      {'nOut': 6, 'activation': 'tanh'},
      {'nOut': 2, 'activation': 'sigmoid'}
    ],
    'decoder': [
      {'nOut': 6, 'activation': 'tanh'},
      {'nOut': X[0].length}
    ]
  })

  ae.fit(X, {
    'batchSize': 100,
    'iterations': 5000,
    'stepSize': 0.01
  })

  t.plan(3)

  const Y1 = ae.encode(X)
  const Y2 = ae.decode(Y1)
  const Y3 = ae.predict(X)

  t.equal(Y1[0].length, 2)
  t.equal(Y2[0].length, X[0].length)
  t.equal(Y3[0].length, X[0].length)

  console.log(X[0], Y1[0], Y2[0], Y3[0])
})
