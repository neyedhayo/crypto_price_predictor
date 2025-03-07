package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"sort"
	"strconv"
	"time"

	"github.com/go-resty/resty/v2"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	ctx, ch := FetchMarketCandles(context.Background(), "DOGE-USDT-SWAP", time.Now().AddDate(0, 0, -1), time.Now(), CandleBar1m)

	var candles []Candle
outer:
	for {
		select {
		case candle, ok := <-ch:
			if !ok {
				break outer
			}
			candles = append(candles, candle)
		case <-ctx.Done():
			if !errors.Is(ctx.Err(), context.Canceled) {
				log.Fatalf("Context error: %v", ctx.Err())
			}
			break outer
		}
	}

	features, labels := PrepareDataForGorgonia(candles, GorgoniaParams{WindowSize: 13, StrategyLong: 0.2 / 50, StrategyShort: 0.2 / 50, StrategyHold: 0.1 / 50})
	countTraining := int(float64(len(features)) * 0.8)
	trainingFeatures := features[:countTraining]
	trainingLabels := labels[:countTraining]
	testingFeatures := features[countTraining:]
	testingLabels := labels[countTraining:]

	if weights, err := BuildAndTrainNN(trainingFeatures, trainingLabels, 100); err != nil {
		log.Fatalf("Training error: %v", err)
	} else {
		for i := range len(testingFeatures) {
			if value, err := Predict(weights, testingFeatures[i]); err != nil {
				log.Fatalf("Prediction error: %v", err)
			} else {
				predictedClass := argmax(value)
				log.Printf("Prediction: %v, Actual: %v", predictedClass, int(testingLabels[i]))
			}
		}
	}
}

func argmax(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]
	for i, value := range slice {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}
	return maxIndex
}

type CandleData [][]string

type Candle struct {
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

type CandleBar string

const (
	CandleBar1s  CandleBar = "1s"
	CandleBar1m  CandleBar = "1m"
	CandleBar5m  CandleBar = "5m"
	CandleBar15m CandleBar = "15m"
	CandleBar1h  CandleBar = "1h"
)

func CandleBarToDuration(bar CandleBar) time.Duration {
	switch bar {
	case CandleBar1s:
		return time.Second
	case CandleBar1m:
		return time.Minute
	case CandleBar5m:
		return 5 * time.Minute
	case CandleBar15m:
		return 15 * time.Minute
	case CandleBar1h:
		return time.Hour
	default:
		return time.Minute
	}
}

func NewCandlesFromData(data [][]string) ([]Candle, error) {
	out := make([]Candle, len(data))

	for i, candle := range data {
		if len(candle) < 6 {
			return nil, fmt.Errorf("invalid candle data: %v", candle)
		}

		if timestamp, err := strconv.ParseInt(candle[0], 10, 64); err != nil {
			return nil, err
		} else if open, err := strconv.ParseFloat(candle[1], 64); err != nil {
			return nil, err
		} else if high, err := strconv.ParseFloat(candle[2], 64); err != nil {
			return nil, err
		} else if low, err := strconv.ParseFloat(candle[3], 64); err != nil {
			return nil, err
		} else if close, err := strconv.ParseFloat(candle[4], 64); err != nil {
			return nil, err
		} else if volume, err := strconv.ParseFloat(candle[5], 64); err != nil {
			return nil, err
		} else {
			out[i] = Candle{
				Timestamp: time.UnixMilli(timestamp),
				Open:      open,
				High:      high,
				Low:       low,
				Close:     close,
				Volume:    volume,
			}
		}
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].Timestamp.Before(out[j].Timestamp)
	})

	return out, nil
}

func FetchMarketCandles(ctx context.Context, instrument string, start time.Time, end time.Time, bar CandleBar) (context.Context, chan Candle) {
	client := resty.New()
	candles := 50
	out := make(chan Candle, candles*100)
	ctx, cancel := context.WithCancelCause(ctx)

	url := "https://www.okx.com/api/v5/market/history-candles"

	go func() {
		defer close(out)
		defer cancel(nil)

		duration := time.Duration(candles) * CandleBarToDuration(bar)

		for ; start.Before(end); start = start.Add(duration) {
			params := map[string]string{
				"instId": instrument,
				"bar":    string(bar),
				"limit":  fmt.Sprintf("%d", candles),
				"after":  fmt.Sprintf("%d", start.Add(duration).UnixMilli()),
				"before": fmt.Sprintf("%d", start.UnixMilli()),
			}

			log.Printf("fetching candles from: %s to: %s", start, start.Add(duration))

			requested := time.Now()

			if resp, err := client.R().SetContext(ctx).SetQueryParams(params).Get(url); err != nil {
				cancel(err)
				return
			} else if resp.IsError() {
				cancel(fmt.Errorf("error response: %v", resp.Status()))
				return
			} else {
				var data struct {
					Code string     `json:"code"`
					Msg  string     `json:"msg"`
					Data CandleData `json:"data"`
				}

				if err := json.Unmarshal(resp.Body(), &data); err != nil {
					cancel(fmt.Errorf("failed to parse response body: %s", err))
					return
				} else if data.Code != "0" {
					cancel(fmt.Errorf("API Error: %s", data.Msg))
					return
				} else if candles, err := NewCandlesFromData(data.Data); err != nil {
					cancel(fmt.Errorf("failed to convert data to candles: %s", err))
					return
				} else {
					for _, candle := range candles {
						select {
						case out <- candle:
						case <-ctx.Done():
							return
						}
					}
				}
			}

			time.Sleep(time.Until(requested.Add(200 * time.Millisecond)))
		}
	}()

	return ctx, out
}

func MovingAverage(prices []float64, window int) []float64 {
	ma := make([]float64, len(prices))
	for i := range prices {
		if i < window {
			ma[i] = 0
			continue
		}
		sum := 0.0
		for j := 0; j < window; j++ {
			sum += prices[i-j]
		}
		ma[i] = sum / float64(window)
	}
	return ma
}

func RSI(prices []float64, window int) []float64 {
	rsi := make([]float64, len(prices))
	for i := range prices {
		if i < window {
			continue
		}
		gains, losses := 0.0, 0.0
		for j := 0; j < window; j++ {
			change := prices[i-j] - prices[i-j-1]
			if change > 0 {
				gains += change
			} else {
				losses -= change
			}
		}
		avgGain := gains / float64(window)
		avgLoss := losses / float64(window)
		if avgLoss == 0 {
			rsi[i] = 100
		} else {
			rs := avgGain / avgLoss
			rsi[i] = 100 - (100 / (1 + rs))
		}
	}
	return rsi
}

func MACD(prices []float64, shortWindow, longWindow, signalWindow int) ([]float64, []float64) {
	shortMA := MovingAverage(prices, shortWindow)
	longMA := MovingAverage(prices, longWindow)
	macd := make([]float64, len(prices))
	var signal []float64

	for i := range prices {
		macd[i] = shortMA[i] - longMA[i]
	}
	signal = MovingAverage(macd, signalWindow)

	return macd, signal
}

func BollingerBands(prices []float64, window int, multiplier float64) ([]float64, []float64, []float64) {
	ma := MovingAverage(prices, window)
	upper, lower := make([]float64, len(prices)), make([]float64, len(prices))

	for i := range prices {
		if i < window {
			upper[i], lower[i] = 0, 0
			continue
		}
		sum := 0.0
		for j := i - window + 1; j <= i; j++ {
			sum += math.Pow(prices[j]-ma[i], 2)
		}
		stdDev := math.Sqrt(sum / float64(window))
		upper[i] = ma[i] + multiplier*stdDev
		lower[i] = ma[i] - multiplier*stdDev
	}
	return ma, upper, lower
}

func StochasticOscillator(closes, lows, highs []float64, window int) ([]float64, []float64) {
	kValues := make([]float64, len(closes))
	dValues := make([]float64, len(closes))

	for i := range closes {
		if i < window {
			kValues[i], dValues[i] = 0, 0
			continue
		}
		low, high := lows[i], highs[i]
		for j := i - window + 1; j <= i; j++ {
			low = math.Min(low, lows[j])
			high = math.Max(high, highs[j])
		}
		kValues[i] = 100 * (closes[i] - low) / (high - low)
	}

	// Calculate %D as a 3-period moving average of %K
	dValues = MovingAverage(kValues, 3)

	return kValues, dValues
}

func VWAP(closes, volumes []float64) []float64 {
	vwap := make([]float64, len(closes))
	cumulativeVolume, cumulativeValue := 0.0, 0.0

	for i := range closes {
		cumulativeVolume += volumes[i]
		cumulativeValue += closes[i] * volumes[i]
		if cumulativeVolume != 0 {
			vwap[i] = cumulativeValue / cumulativeVolume
		} else {
			vwap[i] = 0
		}
	}
	return vwap
}

type Strategy float64

const (
	StrategyHold  Strategy = 0
	StrategyLong  Strategy = 1
	StrategyShort Strategy = 2
)

type GorgoniaParams struct {
	WindowSize    int
	StrategyLong  float64
	StrategyShort float64
	StrategyHold  float64
}

func PrepareDataForGorgonia(candles []Candle, params GorgoniaParams) ([][]float64, []float64) {
	features := [][]float64{}
	labels := []float64{}

	closes := make([]float64, len(candles))
	lows := make([]float64, len(candles))
	highs := make([]float64, len(candles))
	volumes := make([]float64, len(candles))

	for i, candle := range candles {
		closes[i] = candle.Close
		lows[i] = candle.Low
		highs[i] = candle.High
		volumes[i] = candle.Volume
	}

	ma50 := MovingAverage(closes, 50)
	ma200 := MovingAverage(closes, 200)
	rsi14 := RSI(closes, 14)
	macd, macdSignal := MACD(closes, 12, 26, 9)
	ma20, bbUpper, bbLower := BollingerBands(closes, 20, 2.0)
	stochK, stochD := StochasticOscillator(closes, lows, highs, 14)
	vwap := VWAP(closes, volumes)

	for i := params.WindowSize; i < len(candles)-5; i++ {
		window := candles[i-params.WindowSize : i]

		// Feature extraction
		feature := []float64{
			window[params.WindowSize-1].Close,  // Latest close price
			ma50[i],                            // 50-period MA
			ma200[i],                           // 200-period MA
			rsi14[i],                           // 14-period RSI
			macd[i],                            // MACD
			macdSignal[i],                      // MACD Signal line
			ma20[i],                            // 20-period MA (Bollinger Middle Band)
			bbUpper[i],                         // Bollinger Upper Band
			bbLower[i],                         // Bollinger Lower Band
			stochK[i],                          // Stochastic %K
			stochD[i],                          // Stochastic %D
			vwap[i],                            // Volume Weighted Average Price
			window[params.WindowSize-1].Volume, // Latest volume
		}
		features = append(features, feature)

		// Labeling strategy:
		// - StrategyLong if 40% gain within next 5 candles
		// - StrategyShort if 40% loss within next 5 candles
		// - StrategyHold if a loss is avoided
		label := StrategyHold
		basePrice := candles[i].Close
		low := candles[i].Low
		high := candles[i].High
		for j := 1; j <= 5; j++ {
			low = math.Min(low, candles[i+j].Low)
			high = math.Max(high, candles[i+j].High)
			if priceChange := (candles[i+j].High - basePrice) / basePrice; priceChange >= params.StrategyLong {
				if priceChange := (low - basePrice) / basePrice; priceChange > -params.StrategyHold {
					label = StrategyLong
				}
				break
			} else if priceChange := (candles[i+j].Low - basePrice) / basePrice; priceChange <= -params.StrategyShort {
				if priceChange := (high - basePrice) / basePrice; priceChange > params.StrategyHold {
					label = StrategyShort
				}
				break
			}
		}
		labels = append(labels, float64(label))
	}

	return NormalizeData(features), labels
}

func NormalizeData(features [][]float64) [][]float64 {
	for i := range features[0] {
		min, max := math.Inf(1), math.Inf(-1)
		for j := range features {
			min = math.Min(min, features[j][i])
			max = math.Max(max, features[j][i])
		}
		if max > min {
			for j := range features {
				features[j][i] = (features[j][i] - min) / (max - min)
			}
		}
	}
	return features
}

func CategoricalCrossEntropy(pred, target *gorgonia.Node) (*gorgonia.Node, error) {
	logPred, err := gorgonia.Log(pred)
	if err != nil {
		return nil, err
	}
	ce, err := gorgonia.HadamardProd(target, logPred)
	if err != nil {
		return nil, err
	}
	meanCE, err := gorgonia.Mean(ce)
	if err != nil {
		return nil, err
	}
	return gorgonia.Neg(meanCE)
}

func OneHotEncode(labels []float64, numClasses int) [][]float64 {
	oneHot := make([][]float64, len(labels))
	for i, label := range labels {
		row := make([]float64, numClasses)
		row[int(label)] = 1.0
		oneHot[i] = row
	}
	return oneHot
}

// Flatten the 2D one-hot encoded labels into a 1D slice
func FlattenOneHot(oneHot [][]float64) []float64 {
	flat := make([]float64, 0, len(oneHot)*len(oneHot[0]))
	for _, row := range oneHot {
		flat = append(flat, row...)
	}
	return flat
}

func BuildAndTrainNN(features [][]float64, labels []float64, epochs int) ([]tensor.Tensor, error) {
	g := gorgonia.NewGraph()

	inputSize := len(features[0])
	outputSize := 3
	batchSize := len(features)

	// Input and output tensors
	flatFeatures := make([]float64, batchSize*inputSize)
	for i := 0; i < batchSize; i++ {
		copy(flatFeatures[i*inputSize:(i+1)*inputSize], features[i])
	}

	// Explicitly set the tensor shape to avoid shape mismatch
	xVal := tensor.New(
		tensor.WithShape(batchSize, inputSize),
		tensor.Of(tensor.Float64),
		tensor.WithBacking(flatFeatures),
	)

	xTensor := gorgonia.NewTensor(
		g, tensor.Float64, 2,
		gorgonia.WithShape(batchSize, inputSize),
		gorgonia.WithName("x"),
		gorgonia.WithValue(xVal),
	)

	oneHotLabels := OneHotEncode(labels, 3) // Assuming 3 classes: 0, 1, 2
	flatLabels := FlattenOneHot(oneHotLabels)
	yVal := tensor.New(
		tensor.WithShape(batchSize, outputSize),
		tensor.Of(tensor.Float64),
		tensor.WithBacking(flatLabels),
	)

	yTensor := gorgonia.NewTensor(
		g, tensor.Float64, 2,
		gorgonia.WithShape(batchSize, outputSize),
		gorgonia.WithName("y"),
		gorgonia.WithValue(yVal),
	)

	// Weight and bias initialization with gradient tracking
	w0 := gorgonia.NewMatrix(
		g, tensor.Float64,
		gorgonia.WithShape(inputSize, 10),
		gorgonia.WithName("w0"),
		gorgonia.WithInit(gorgonia.GlorotU(1)),
		gorgonia.WithGrad(tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(inputSize, 10))),
	)

	b0 := gorgonia.NewMatrix(
		g, tensor.Float64,
		gorgonia.WithShape(1, 10), // Bias is (1, 10) for correct broadcasting
		gorgonia.WithName("b0"),
		gorgonia.WithInit(gorgonia.Zeroes()),
		gorgonia.WithGrad(tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(1, 10))),
	)

	w1 := gorgonia.NewMatrix(
		g, tensor.Float64,
		gorgonia.WithShape(10, outputSize),
		gorgonia.WithName("w1"),
		gorgonia.WithInit(gorgonia.GlorotU(1)),
		gorgonia.WithGrad(tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(10, outputSize))),
	)

	b1 := gorgonia.NewMatrix(
		g, tensor.Float64,
		gorgonia.WithShape(1, outputSize), // Bias is (1, outputSize) for broadcasting
		gorgonia.WithName("b1"),
		gorgonia.WithInit(gorgonia.Zeroes()),
		gorgonia.WithGrad(tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(1, outputSize))),
	)

	// Forward pass with bias broadcasting
	l0Raw := gorgonia.Must(gorgonia.Mul(xTensor, w0))
	l0 := gorgonia.Must(gorgonia.BroadcastAdd(l0Raw, b0, nil, []byte{0}))
	l0Act := gorgonia.Must(gorgonia.LeakyRelu(l0, 0.01)) // ReLU activation

	predRaw := gorgonia.Must(gorgonia.Mul(l0Act, w1))
	pred := gorgonia.Must(gorgonia.BroadcastAdd(predRaw, b1, nil, []byte{0}))
	predAct := gorgonia.Must(gorgonia.SoftMax(pred)) // Softmax for multi-class classification

	// Binary Cross Entropy Loss
	loss, err := CategoricalCrossEntropy(predAct, yTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to compute binary cross entropy: %w", err)
	}

	gorgonia.WithName("loss")(loss)
	gorgonia.WithName("l0Raw")(l0Raw)
	gorgonia.WithName("predAct")(predAct)

	// Create a virtual machine and bind dual values automatically
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(w0, b0, w1, b1, xTensor, yTensor, loss))
	defer vm.Close()

	// Prepare input data
	// Prepare input data with explicit reshaping

	learningRate := 0.001

	for epoch := 0; epoch < epochs; epoch++ {
		vm.Reset()

		gorgonia.Let(xTensor, xVal)
		gorgonia.Let(yTensor, yVal)

		if err := vm.RunAll(); err != nil {
			return nil, fmt.Errorf("error during training: %w", err)
		}

		solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(learningRate))
		if err := solver.Step([]gorgonia.ValueGrad{w0, b0, w1, b1}); err != nil {
			return nil, fmt.Errorf("error during solver step: %w", err)
		}

		gradW0, _ := w0.Grad()
		gradB0, _ := b0.Grad()
		gradW1, _ := w1.Grad()
		gradB1, _ := b1.Grad()

		fmt.Printf("Gradients - w0: %v, b0: %v, w1: %v, b1: %v\n", gradW0, gradB0, gradW1, gradB1)

		fmt.Printf("Epoch %d - Loss: %v\n", epoch, loss.Value())
	}

	for _, n := range g.AllNodes() {
		grad, _ := n.Grad()
		fmt.Printf("Node: %s, Op: %v, Has Value: %v, Has Gradient: %v\n",
			n.Name(), n.Op(), n.Value() != nil, grad != nil)
	}

	w0Val, _ := w0.Value().(tensor.Tensor)
	b0Val, _ := b0.Value().(tensor.Tensor)
	w1Val, _ := w1.Value().(tensor.Tensor)
	b1Val, _ := b1.Value().(tensor.Tensor)

	return []tensor.Tensor{w0Val, b0Val, w1Val, b1Val}, nil
}

func Predict(weights []tensor.Tensor, input []float64) ([]float64, error) {
	g := gorgonia.NewGraph()
	inputSize := len(input)

	// Input tensor
	xVal := tensor.New(
		tensor.WithShape(1, inputSize),
		tensor.Of(tensor.Float64),
		tensor.WithBacking(input),
	)
	xTensor := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(1, inputSize), gorgonia.WithName("input"), gorgonia.WithValue(xVal))

	// Load weights as constants
	w0 := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(weights[0].Shape()...), gorgonia.WithValue(weights[0]))
	b0 := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(weights[1].Shape()...), gorgonia.WithValue(weights[1]))
	w1 := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(weights[2].Shape()...), gorgonia.WithValue(weights[2]))
	b1 := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(weights[3].Shape()...), gorgonia.WithValue(weights[3]))

	// Forward pass with bias broadcasting
	l0Raw := gorgonia.Must(gorgonia.Mul(xTensor, w0))
	l0 := gorgonia.Must(gorgonia.BroadcastAdd(l0Raw, b0, nil, []byte{0}))
	l0Act := gorgonia.Must(gorgonia.LeakyRelu(l0, 0.01))

	predRaw := gorgonia.Must(gorgonia.Mul(l0Act, w1))
	pred := gorgonia.Must(gorgonia.BroadcastAdd(predRaw, b1, nil, []byte{0}))
	predAct := gorgonia.Must(gorgonia.SoftMax(pred))

	// Create VM to run the graph
	vm := gorgonia.NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		return nil, err
	}

	return predAct.Value().Data().([]float64)[0:3], nil
}
