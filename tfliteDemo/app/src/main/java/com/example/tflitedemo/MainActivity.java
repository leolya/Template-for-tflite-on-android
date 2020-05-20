package com.example.tflitedemo;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collections;
import java.util.Vector;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();
    private static final int PICTURE_FROM_GALLERY = 101;
    private static final String MODEL = "mobilenet.tflite"; // model name

    private ImageView imageV;
    private TextView textV;
    private TextView textV_C;
    private Bitmap bitmap;
    private Interpreter tflite;
    private  long inferenceTime = 0;
    private Vector<String> labels = new Vector<String>();
    private AssetManager assetManager;
    private String prediction;

    Handler mHandler = new Handler(){
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            if(msg.arg1 == 1 && inferenceTime != 0) {
                textV.setText("inference time: " + inferenceTime);
                textV_C.setText("Prediction: " + prediction);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageV = (ImageView) findViewById(R.id.imageView);
        textV = (TextView) findViewById(R.id.textView);
        textV_C = (TextView) findViewById(R.id.textView_label);

        new ModelLoadAsyncTask().execute();

        findViewById(R.id.btnPickImage).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                choosePhotoFromGallery();
            }
        });

        findViewById(R.id.btnPredict).setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                if(bitmap != null && tflite != null){
                    new ModelRunAsyncTask().execute();
                }
            }
        });

        assetManager = getAssets();  // used to load label from assets folder
    }

    private class ModelLoadAsyncTask extends AsyncTask<Void, Void, Integer> {
        @Override
        protected Integer doInBackground(Void... args) {
            try {
                tflite = new Interpreter(loadModelFile(MODEL));
            } catch (IOException e) {
                return 0;
            }

            //load labels
            try {
                String labelsContent = new String(getBytesFromFile(assetManager, "imagenet_labels.list"));
                labels.addAll(Arrays.asList(labelsContent.split("\r")));
            } catch (IOException e) {
                return 0;//failure
            }

            return 1;
        }
        protected void onPostExecute(Integer status) {
            if (status != 1) {
                Log.e(TAG, "Load model failed!!");
            }
        }
    }

    private class ModelRunAsyncTask extends AsyncTask<Void, Void, Integer> {
        @Override
        protected Integer doInBackground(Void... args) {
//            int width = bitmap.getWidth();
//            int height = bitmap.getHeight();
//            float[][][][] out = new float[1][height][width][3];

            int height = 224;
            int width = 224;
            float[][]out = new float[1][1000];
            float[][] norm = {{123.f,117.f,104.f},{58.395f,57.12f,57.375f}};
            float[][][][] in = getScaledMatrix(height, width, bitmap, norm);
            long startTimeForInference = SystemClock.uptimeMillis();
            tflite.run(in, out);
            long endTimeForInference = SystemClock.uptimeMillis();
            inferenceTime = endTimeForInference - startTimeForInference;
            Log.i(TAG, "inference Time: " + inferenceTime);


            int maxPosition = -1;
            float maxValue = 0;
            for (int j = 0; j < out[0].length; ++j) {
                if (out[0][j] > maxValue) {
                    maxValue = out[0][j];
                    maxPosition = j;
                }
            }
            prediction = labels.get(maxPosition);

//            outputBitmap = getBitmap(height, width, out);
            Message msg = Message.obtain();
            msg.arg1 = 1;
            mHandler.sendMessage(msg);
            return 1;
        }
    }

    private MappedByteBuffer loadModelFile(String model) throws IOException {
        AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd(model);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private byte[] getBytesFromFile(AssetManager assets, String fileName) throws IOException {
        InputStream is = assets.open(fileName);
        int length = is.available();
        byte[] bytes = new byte[length];
        // Read in the bytes
        int offset = 0;
        int numRead = 0;
        try {
            while (offset < bytes.length
                    && (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
                offset += numRead;
            }
        } finally {
            is.close();
        }
        // Ensure all the bytes have been read in
        if (offset < bytes.length) {
            throw new IOException("Could not completely read file " + fileName);
        }
        return bytes;
    }

    // convert the input bitmap into a float array
    public float[][][][] getScaledMatrix(int height, int width, Bitmap bitmap, float[][] norm) {
        float[][][][] inFloat = new float[1][height][width][3];
        int[] pixels = new int[height * width];
        Bitmap bm = Bitmap.createScaledBitmap(bitmap, width, height, true);
        bm.getPixels(pixels, 0, width, 0, 0, width, height);
        int pixel = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                final int val = pixels[pixel++];
                float red = (((val >> 16) & 0xFF)-norm[0][0])/norm[1][0];
                float green = (((val >> 8) & 0xFF)-norm[0][1])/norm[1][1];
                float blue = ((val & 0xFF)-norm[0][2])/norm[1][2];
                float[] arr = {red, green, blue};
                inFloat[0][i][j] = arr;
            }
        }
        return inFloat;
    }

    // convert the output float array into bitmap
    public Bitmap getBitmap(int height, int width, float[][][][] outArr) {
        int n = 0;
        int[] colorArr = new int[height * width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float[] arr = outArr[0][i][j];
                if(arr[0]>255){arr[0]=255;}
                else if(arr[0]<0){arr[0]=0;}
                if(arr[1]>255){arr[1]=255;}
                else if(arr[1]<0){arr[1]=0;}
                if(arr[2]>255){arr[2]=255;}
                else if(arr[2]<0){arr[2]=0;}
                int alpha = 255;
                int red = (int) arr[0];
                int green = (int) arr[1];
                int blue = (int) arr[2];
                int tempARGB = (alpha << 24) | (red << 16) | (green << 8) | blue;
                colorArr[n++] = tempARGB;
            }
        }
        return Bitmap.createBitmap(colorArr, width, height, Bitmap.Config.ARGB_8888);
    }


    public void choosePhotoFromGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, PICTURE_FROM_GALLERY);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == this.RESULT_CANCELED) {
            return;
        }
        Uri contentURI = null;
        if (requestCode == PICTURE_FROM_GALLERY) {
            if (data != null) {
                contentURI = data.getData();
            }
            if (null != contentURI) {
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), contentURI);
                    imageV.setScaleType(ImageView.ScaleType.FIT_CENTER);
                    imageV.setImageBitmap(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

}
