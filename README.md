# Template-for-tflite-on-android

This is a template to test the reference speed of tflite model on android.



- run the notebook to generate tflite model

  ```python
  import tensorflow as tf
  from tensorflow.keras.applications import MobileNetV2
  
  model = MobileNetV2(weights='imagenet')
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  open("mobilenet.tflite", "wb").write(tflite_model)
  ```

- put tflite model in assets folder

  ".\tfliteDemo\app\src\main\assets\"

- change model name in MainActivity.java

  ```java
  private static final String MODEL = "mobilenet.tflite"; // model name
  ```

- change the input and output shape in MainActivity.java "ModelRunAsyncTask"

- you can put tflite.run(in, out) in a loop to test average reference speed

  ```java
      private class ModelRunAsyncTask extends AsyncTask<Void, Void, Integer> {
          @Override
          protected Integer doInBackground(Void... args) {
  //            if output is an image
  //            int width = bitmap.getWidth();
  //            int height = bitmap.getHeight();
  //            float[][][][] out = new float[1][height][width][3];
              
              int height = 224;  // input shape
              int width = 224;
              float[][]out = new float[1][1000];  // output array
              float[][][][] in = getScaledMatrix(height, width, bitmap);
              long startTimeForReference = SystemClock.uptimeMillis();
              tflite.run(in, out);
              long endTimeForReference = SystemClock.uptimeMillis();
              referenceTime = endTimeForReference - startTimeForReference;
              Log.i(TAG, "Reference Time: " + referenceTime);
  
  //            outputBitmap = getBitmap(height, width, out);
              Message msg = Message.obtain();
              msg.arg1 = 1;
              mHandler.sendMessage(msg);
              return 1;
          }
      }
  ```



https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#0