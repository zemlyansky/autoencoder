const Tensor = require('adnn/tensor')
const nn = require('adnn/nn')
const opt = require('adnn/opt')

module.exports = class Autoencoder {
  constructor (params) {
    this.scale = params.scale || true

    let encoderLayers = []
    let decoderLayers = []

    if (params.encoder && params.encoder.length && params.decoder && params.decoder.length) {
      this.nInputs = params.decoder[params.decoder.length - 1]['nOut']
      this.nHidden = params.encoder[params.encoder.length - 1]['nOut']
      encoderLayers = params.encoder.map(layer => ({'nOut': layer.nOut, 'activation': nn[layer.activation]}))
      decoderLayers = params.decoder.map(layer => ({'nOut': layer.nOut, 'activation': nn[layer.activation]}))
    } else if (params.nInputs && params.nHidden) {
      this.nInputs = params.nInputs
      this.nHidden = params.nHidden
      const nLayers = params.nLayers || 2
      const activation = params.activation || 'relu'
      for (var i = 0; i < nLayers; i++) {
        encoderLayers.push({
          'nOut': this.nInputs - Math.round((i + 1) * (this.nInputs - this.nHidden) / nLayers),
          'activation': nn[activation]
        })
        decoderLayers.push({
          'nOut': this.nHidden + Math.round((i + 1) * (this.nInputs - this.nHidden) / nLayers),
          'activation': (i < nLayers - 1) ? nn[activation] : undefined
        })
      }
    } else {
      throw new Error('Not enough parameters to build an autoencoder')
    }

    this.encoder = nn.mlp(this.nInputs, encoderLayers)
    this.decoder = nn.mlp(this.nHidden, decoderLayers)
    this.net = nn.sequence([
      this.encoder,
      this.decoder
    ])
  }

  _scalerFit (X) {
    this.max = X[0].slice(0)
    this.min = X[0].slice(0)
    X.forEach(x => {
      x.forEach((v, i) => {
        if (v > this.max[i]) {
          this.max[i] = v
        } else if (v < this.min[i]) {
          this.min[i] = v
        }
      })
    })
  }

  // xs = (x - min)/(max - min)
  _scalerTransform (X) {
    const diff = this.max.map((max, i) => max - this.min[i])
    return X.map(x => x.map((v, i) => {
      return diff[i] > 0
        ? (v - this.min[i]) / (this.max[i] - this.min[i])
        : 0
    }))
  }

  _scalerFitTransform (X) {
    this._scalerFit(X)
    return this._scalerTransform(X)
  }

  // x = min + xs * (max - min)
  _scalerInverseTransform (X) {
    return X.map(x => x.map((v, i) => this.min[i] + v * (this.max[i] - this.min[i])))
  }

  fit (X, p) {
    const params = p || {}
    const stepSize = params.stepSize || 0.05
    const Xs = this.scale ? this._scalerFitTransform(X) : X
    const Xt = Xs.map(x => {
      const t = (new Tensor([this.nInputs])).fromArray(x)
      return {'input': t, 'output': t}
    })
    opt.nnTrain(this.net, Xt, opt.regressionLoss, {
      iterations: params.iterations || 100,
      batchSize: params.batchSize || Math.round(Xt.length / 50),
      method: params.method ? opt[params.method]({stepSize}) : opt.adagrad({stepSize})
    })
  }

  _eval (X, model) {
    const n = X[0].length
    return X.map(x => model.eval((new Tensor([n])).fromArray(x)).toArray())
  }

  encode (X) {
    const Xs = this.scale ? this._scalerTransform(X) : X
    return this._eval(Xs, this.encoder)
  }

  decode (Y) {
    const Xout = this._eval(Y, this.decoder)
    return this.scale ? this._scalerInverseTransform(Xout) : Xout
  }

  predict (X) {
    const Xs = this.scale ? this._scalerTransform(X) : X
    const Xout = this._eval(Xs, this.net)
    return this.scale ? this._scalerInverseTransform(Xout) : Xout
  }
}
