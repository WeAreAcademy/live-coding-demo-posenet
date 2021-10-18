let videoCapture;

let results;

function setup() {
  setupVideoCapture();
  createCanvas(videoCapture.width, videoCapture.height);
  setupPoseNet();

  noStroke();
  textSize(10);
}

function draw() {
  // maybe the first results aren't in yet or there are NO guesses
  if (results && results.length > 0) {
    let skelColour = "lime";
    if (isRaisedHand(results[0].pose)) {
      // background("pink");
      image(videoCapture, 0, 0);
      skelColour = "red";
    } else {
      background(255);
    }
    drawKeypoints(results[0].pose);
    drawSkeleton(results[0].skeleton, skelColour);
  }
}

function modelLoaded() {
  console.log("Model Loaded!");
}

function storeNewPoseResults(data) {
  results = data;
}

function drawSkeleton(connections, colour) {
  //for each CONNECTION_PAIR in CONNECTIONS
  //   let A = CONNECTION_PAIR[0]
  //   let B = CONNECTION_PAIR[1]
  //   draw line from A's position to B's position
  stroke(colour);
  strokeWeight(3);
  for (let [a, b] of connections) {
    line(a.position.x, a.position.y, b.position.x, b.position.y);
  }
}

function drawKeypoints(pose) {
  const probableKeypoints = pose.keypoints.filter((kp) => kp.score > 0.6);
  noStroke();

  for (let keypoint of probableKeypoints) {
    const { x, y } = keypoint.position;
    fill("gray");
    if (keypoint.part === "leftEye" || keypoint.part === "rightEye") {
      drawEye(x, y);
    } else {
      square(x, y, 10);
    }
    noStroke();
    fill("black");
    text(
      keypoint.part + "\n" + keypoint.score.toFixed(2),
      keypoint.position.x,
      keypoint.position.y + 30
    );
  }
}

function drawEye(x, y) {
  stroke("black");
  fill("white");
  circle(x, y, 20);
  fill("black");
  circle(x, y, 10);
}

function isRaisedHand(pose) {
  return pose.leftWrist.y < pose.nose.y && pose.leftWrist.confidence > 0.5;
}

function setupVideoCapture() {
  videoCapture = createCapture(VIDEO);
  videoCapture.size(400, 300);
  videoCapture.hide();
}

function setupPoseNet() {
  // Create a new poseNet method
  const poseNet = ml5.poseNet(videoCapture, "single", modelLoaded);
  // Listen to new 'pose' events
  poseNet.on("pose", storeNewPoseResults);
}
