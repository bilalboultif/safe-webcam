const express = require('express');
const multer = require('multer');
const jpeg = require('jpeg-js');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const nsfw = require('nsfwjs');
const ffmpeg = require('fluent-ffmpeg');
const path = require('path');
const fs = require('fs');
const { promisify } = require('util');
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args)); // Dynamic import
const app = express();
const upload = multer({ storage: multer.memoryStorage() });

let _model;

// Convert image buffer to tensor
const convert = async (img) => {
  const image = jpeg.decode(img, true);
  const numChannels = 3;
  const numPixels = image.width * image.height;
  const values = new Int32Array(numPixels * numChannels);

  for (let i = 0; i < numPixels; i++) {
    for (let c = 0; c < numChannels; ++c) {
      values[i * numChannels + c] = image.data[i * 4 + c];
    }
  }

  return tf.tensor3d(values, [image.height, image.width, numChannels], 'int32');
};

// Extract frames from video at specified timestamps
const extractFramesFromVideo = (videoPath, timestamps, outputDir) => {
  return new Promise((resolve, reject) => {
    console.log(`Extracting frames from ${videoPath}...`);
    ffmpeg(videoPath)
      .on('end', () => resolve())
      .on('error', (err) => reject(err))
      .screenshots({
        timestamps: timestamps,
        filename: 'frame-%03d.jpg',
        folder: outputDir,
        size: '640x?'
      });
  });
};

// Apply blur effect to frames that are flagged as unsafe
const applyBlurToFrame = (inputPath, outputPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .videoFilters('boxblur=10:1')
      .save(outputPath)
      .on('end', () => resolve())
      .on('error', (err) => reject(err));
  });
};

// Apply blur to unsafe frames and rebuild the video
const processVideo = async (videoPath, frameTimes) => {
  const tempDir = path.join(__dirname, 'temp_frames');
  fs.mkdirSync(tempDir, { recursive: true });

  // Extract frames
  await extractFramesFromVideo(videoPath, frameTimes, tempDir);

  const frames = fs.readdirSync(tempDir).filter(file => file.endsWith('.jpg'));
  const unsafeFrames = [];
  const processedFrames = [];

  // Analyze each frame
  for (const frame of frames) {
    const framePath = path.join(tempDir, frame);
    const frameBuffer = fs.readFileSync(framePath);
    const image = await convert(frameBuffer);
    const predictions = await _model.classify(image);
    image.dispose();

    // Determine if frame is unsafe
    const unsafe = predictions.some(prediction => 
      prediction.className === 'Sexy' && prediction.probability > 0.3 
    );

    if (unsafe) {
      unsafeFrames.push(framePath);
    } else {
      processedFrames.push(framePath);
    }
  }

  // Apply blur to unsafe frames
  for (const framePath of unsafeFrames) {
    const blurredPath = framePath.replace('.jpg', '-blurred.jpg');
    await applyBlurToFrame(framePath, blurredPath);
    processedFrames.push(blurredPath);
  }

  // Rebuild video from processed frames
  const outputVideoPath = path.join(__dirname, 'processed_video.mp4');
  await new Promise((resolve, reject) => {
    ffmpeg()
      .input(path.join(tempDir, 'frame-%03d.jpg'))
      .inputFormat('image2')
      .output(outputVideoPath)
      .on('end', () => {
        fs.rmdirSync(tempDir, { recursive: true });
        resolve();
      })
      .on('error', (err) => reject(err))
      .run();
  });

  return outputVideoPath;
};

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});
app.use(express.static('public'));

app.get('/', (req, res) => {
  res.send('Welcome to the NSFW detection server');
});

app.post('/nsfw', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).send('Missing image multipart/form-data');
  }

  try {
    const image = await convert(req.file.buffer);
    const predictions = await _model.classify(image);
    image.dispose();
    console.log('Image analysis result:', predictions);
    res.json(predictions);
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).send('Error processing image');
  }
});

app.post('/video', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).send('Missing video multipart/form-data');
  }

  const videoPath = path.join(__dirname, 'temp_video.mp4');
  fs.writeFileSync(videoPath, req.file.buffer);

  try {
    // Specify the timestamps for frame extraction (e.g., every 5 seconds)
    const frameTimes = [1, 5, 10, 15, 20]; // Adjust as needed

    const processedVideoPath = await processVideo(videoPath, frameTimes);

    // Serve the processed video
    res.sendFile(processedVideoPath, () => {
      // Clean up temporary files after serving
      fs.unlinkSync(videoPath);
      fs.unlinkSync(processedVideoPath);
    });
  } catch (error) {
    console.error('Error processing video:', error);
    res.status(500).send('Error processing video');
  }
});

app.post('/video-from-url', express.json(), async (req, res) => {
  const { url } = req.body;

  if (!url) {
    return res.status(400).send('Missing video URL');
  }

  try {
    // Download the video from the URL
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch video');
    }

    const videoBuffer = await response.buffer();
    const videoPath = path.join(__dirname, 'temp_video.mp4');
    fs.writeFileSync(videoPath, videoBuffer);

    // Specify the timestamps for frame extraction (e.g., every 5 seconds)
    const frameTimes = [1, 5, 10, 15, 20]; // Adjust as needed

    const processedVideoPath = await processVideo(videoPath, frameTimes);

    // Serve the processed video
    res.sendFile(processedVideoPath, () => {
      // Clean up temporary files after serving
      fs.unlinkSync(videoPath);
      fs.unlinkSync(processedVideoPath);
    });
  } catch (error) {
    console.error('Error processing video from URL:', error);
    res.status(500).send('Error processing video from URL');
  }
});
// Route to stream video from URL
app.get('/stream', async (req, res) => {
  const { videoUrl } = req.query;

  if (!videoUrl) {
    return res.status(400).send('Missing video URL');
  }

  try {
    // Fetch the video stream
    const response = await fetch(videoUrl);
    if (!response.ok) throw new Error('Failed to fetch video');

    // Set the appropriate headers
    res.setHeader('Content-Type', response.headers.get('content-type'));

    // Pipe the video data directly to the response
    response.body.pipe(res);
  } catch (error) {
    console.error('Error streaming video:', error);
    res.status(500).send('Error streaming video');
  }
});

const load_model = async () => {
  _model = await nsfw.load();
};

load_model().then(() => app.listen(5552, () => console.log('Server running on port 5552')));
