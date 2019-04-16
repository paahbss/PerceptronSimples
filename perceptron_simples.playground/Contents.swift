//: A Cocoa based Playground to present user interface

import AppKit
import PlaygroundSupport
import CreateML

let learninRate = 0.01

func importData() -> MLDataTable? {
    guard let csvData = Bundle.main.url(forResource: "iris", withExtension: "csv") else {return nil}
    var dataTable = MLDataTable()
    do {
        dataTable = try MLDataTable(contentsOf: csvData)
    } catch {
        print("Error in load data")
    }
    let newtargetColumn = dataTable.map { (row) -> Double in
        guard let classIris = row["species"]?.stringValue else {
            fatalError("Missing or invalid columns in row.")
        }
        return classIris == "setosa" ? 1.0 : 0.0
    }
    let bias = [Double](repeating: -1, count: dataTable.rows.count)
    dataTable.addColumn(MLDataColumn(bias), named: "bias")
    dataTable.addColumn(newtargetColumn, named: "target")
    return dataTable
}



func activateFunction(dot: Double) -> Double {
    return dot >= 0 ? 1.0 : 0.0
}

func dotProduct(_ v1: [Double],_  v2: [Double]) -> Double {
    if v1.count != v2.count {
        fatalError("different counts")
    }
    var result: Double = 0.0
    for i in 0...v1.count-1 {
        result += v1[i] * v2[i]
    }
    return result
}

func predict(x: [Double], weights: [Double]) ->  Double{
    let dot = dotProduct(x, weights)
    return activateFunction(dot: dot)
}

func mult(escalar: Double, array: [Double]) -> [Double] {
    return array.map { (num) -> Double in
        num * escalar
    }
}

func sum(_ v1: [Double],_  v2: [Double]) -> [Double] {
    if v1.count != v2.count {
        fatalError("different counts")
    }
    return (zip(v1, v2).map { $0.0 + $0.1 })
}


func trainAndTest(data: MLDataTable, withEpochs epochs: Int, learningRate: Double) -> Double {
    let dataColumns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "bias"]
    let (test, train) = data.randomSplit(by: 0.20, seed: 5)
    
    let testY = test[["target"]]
    let trainY = train[["target"]]
    let testX = test[dataColumns]
    let trainX = train[dataColumns]
    
    var yPredict = 0.0
    var error = 0.0
    var weights = (0...dataColumns.count - 1).map{ _ in Double.random(in: 0...1)}
    print(weights)
    print("---------------TRAIN--------")
    for _ in 0...epochs {
        for (x, y) in zip(trainX.rows, trainY.rows) {
            let xDouble = x.values.compactMap { (value) -> Double? in
                return value.doubleValue
            }
            let yDouble = y.values.compactMap { (value) -> Double? in
                return value.doubleValue
            }
            guard let yReal = yDouble.first else {
                fatalError("different counts")
            }
            yPredict = predict(x: xDouble, weights: weights)
            error = yReal - yPredict
            let firstMult = mult(escalar: error, array: xDouble)
            let secondMult = mult(escalar: learningRate, array: firstMult)
            weights = sum(weights, secondMult)
        }
    }
    print("---------------TEST--------")
    let acurracy = testData(testX: testX, testY: testY, weights: weights)
    return acurracy
}

func testData(testX: MLDataTable, testY: MLDataTable, weights: [Double]) -> Double {
    var yExpected = 0.0
    var error = 0.0
    var hits = 0.0
    for (x, y) in zip(testX.rows, testY.rows) {
        let xDouble = x.values.compactMap { (value) -> Double? in
            return value.doubleValue
        }
        let yDouble = y.values.compactMap { (value) -> Double? in
            return value.doubleValue
        }
        guard let yFirst = yDouble.first else {
            fatalError("different counts")
//            return 0
        }
        yExpected = predict(x: xDouble, weights: weights)
        error = yFirst - yExpected
        if error == 0 {
            hits += 1
        }
    }
    return hits / Double(testY.rows.count)
}

if let data = importData() {
    var acurracyHistoric: [Double] = []
    for _ in 0...5 {
        let ac = trainAndTest(data: data, withEpochs: 100, learningRate: learninRate)
        acurracyHistoric.append(ac)
        print(ac)
    }
    let sum = acurracyHistoric.reduce(0, +)
    print(sum / Double(acurracyHistoric.count))
}
