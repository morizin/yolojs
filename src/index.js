import React from "react";
import ReactDOM from "react-dom";

import "./styles.css";
const tf = require('@tensorflow/tfjs');

const weights = '/web_model/model.json';

const names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();

  state = {
    model: null
  };

  componentDidMount() {
    tf.loadGraphModel(weights).then(model => {
      this.setState({
        model: model
      });
      this.runInference();
    });
  }

  runInference = () => {
    const video = this.videoRef.current;
    const canvas = this.canvasRef.current;
    const ctx = canvas.getContext("2d");

    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        video.play();
        this.processVideoFrame(video, canvas, ctx);
      })
      .catch(error => {
        console.log("Error accessing camera: ", error);
      });
  };

  processVideoFrame = (video, canvas, ctx) => {
    if (!video.paused && !video.ended) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const [modelWidth, modelHeight] = this.state.model.inputs[0].shape.slice(1, 3);
      const input = tf.tidy(() => {
        return tf.image.resizeBilinear(tf.browser.fromPixels(canvas), [modelWidth, modelHeight])
          .div(255.0).expandDims(0);
      });

      this.state.model.executeAsync(input).then(res => {
        // Font options.
        const font = "16px sans-serif";
        ctx.font = font;
        ctx.textBaseline = "top";

        const [boxes, scores, classes, valid_detections] = res;
        const boxesData = boxes.dataSync();
        const scoresData = scores.dataSync();
        const classesData = classes.dataSync();
        const validDetectionsData = valid_detections.dataSync()[0];

        tf.dispose(res);

        for (let i = 0; i < validDetectionsData; ++i) {
          let [x1, y1, x2, y2] = boxesData.slice(i * 4, (i + 1) * 4);
          x1 *= canvas.width;
          x2 *= canvas.width;
          y1 *= canvas.height;
          y2 *= canvas.height;
          const width = x2 - x1;
          const height = y2 - y1;
        //   const klass = names[classesData[i]];
          const klass = "pothole";
          const score = scoresData[i].toFixed(2);
        
          if (score <= 0.15 || klass === 5) {
              continue;
          }

          // Draw the bounding box.
          ctx.strokeStyle = "#00FFFF";
          ctx.lineWidth = 4;
          ctx.strokeRect(x1, y1, width, height);

          // Draw the label background.
          ctx.fillStyle = "#00FFFF";
          const textWidth = ctx.measureText(klass + ":" + score).width;
          const textHeight = parseInt(font, 10); // base 10
          ctx.fillRect(x1, y1, textWidth + 4, textHeight + 4);

          // Draw the text last to ensure it's on top.
          ctx.fillStyle = "#000000";
          ctx.fillText(klass + ":" + score, x1, y1);
        }

        requestAnimationFrame(() => this.processVideoFrame(video, canvas, ctx));
      });
    }
  };

  render() {
    return (
      <div className="Dropzone-page">
        {this.state.model ? (
          <React.Fragment>
            <video
              ref={this.videoRef}
              className="Dropzone-video"
              autoPlay
              muted
            />
            <canvas
              ref={this.canvasRef}
              id="canvas"
              width="640"
              height="480"
              className="Dropzone-canvas"
            />
          </React.Fragment>
        ) : (
          <div className="Dropzone">Loading model...</div>
        )}
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
