import React, { useEffect, useRef } from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs";

import "./styles.css";

const names = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const runObjectDetection = async () => {
      await tf.setBackend("webgl");

      const model = await tf.loadGraphModel("/web_model/model.json");

      const videoElement = videoRef.current;
      const canvasElement = canvasRef.current;
      const context = canvasElement.getContext("2d");

      const handleFrame = () => {
        tf.nextFrame().then(() => {
          context.drawImage(
            videoElement,
            0,
            0,
            videoElement.width,
            videoElement.height
          );

          model.executeAsync(
            tf.browser.fromPixels(videoElement).expandDims()
          )
            .then((result) => {
              const [boxes, scores, classes, numDetections] = result;

              const boxesData = boxes.arraySync()[0];
              const scoresData = scores.arraySync()[0];
              const classesData = classes.arraySync()[0];
              const numDetectionsData = numDetections.arraySync()[0];

              context.clearRect(
                0,
                0,
                canvasElement.width,
                canvasElement.height
              );

              for (let i = 0; i < numDetectionsData; i++) {
                const [y, x, height, width] = boxesData[i];
                const score = scoresData[i];
                const classIndex = classesData[i];

                if (score > 0.5) {
                  const label = names[classIndex];
                  context.beginPath();
                  context.rect(
                    x * videoElement.width,
                    y * videoElement.height,
                    width * videoElement.width,
                    height * videoElement.height
                  );
                  context.lineWidth = 2;
                  context.strokeStyle = "red";
                  context.fillStyle = "red";
                  context.stroke();
                  context.fillText(
                    `${label} (${(score * 100).toFixed(1)}%)`,
                    x * videoElement.width + 5,
                    y * videoElement.height + 16
                  );
                  context.closePath();
                }
              }

              tf.dispose([boxes, scores, classes, numDetections]);

              requestAnimationFrame(handleFrame);
            });
        });
      };

      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          videoElement.srcObject = stream;
          videoElement.onloadedmetadata = () => {
            videoElement.play();
            handleFrame();
          };
        })
        .catch((error) => {
          console.error("Error accessing webcam:", error);
        });
    };

    runObjectDetection();
  }, []);

  return (
    <div className="App">
      <h1>Real-Time Object Detection</h1>
      <div className="CanvasContainer">
        <video className="Video" ref={videoRef} />
        <canvas className="Canvas" ref={canvasRef} />
      </div>
    </div>
  );
};

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
