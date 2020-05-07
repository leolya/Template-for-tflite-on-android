package com.example.tflitedemo;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
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
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();
    private static final int PICTURE_FROM_GALLERY = 101;
    private static final String MODEL = "mobilenet.tflite"; // model name

    private ImageView imageV;
    private TextView textV;
    private Bitmap bitmap;
    private Interpreter tflite;
    private  long referenceTime = 0;

    Handler mHandler = new Handler(){
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            if(msg.arg1 == 1 && referenceTime != 0) {
                textV.setText("reference time: " + referenceTime);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageV = (ImageView) findViewById(R.id.imageView);
        textV = (TextView) findViewById(R.id.textView);

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
    }

    private class ModelLoadAsyncTask extends AsyncTask<Void, Void, Integer> {
        @Override
        protected Integer doInBackground(Void... args) {
            try {
                tflite = new Interpreter(loadModelFile(MODEL));
            } catch (IOException e) {
                return 0;
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

    private MappedByteBuffer loadModelFile(String model) throws IOException {
        AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd(model);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // convert the input bitmap into a float array
    public float[][][][] getScaledMatrix(int height, int width, Bitmap bitmap) {
        float[][][][] inFloat = new float[1][height][width][3];
        int[] pixels = new int[height * width];
        Bitmap bm = Bitmap.createScaledBitmap(bitmap, width, height, true);
        bm.getPixels(pixels, 0, width, 0, 0, width, height);
        int pixel = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                final int val = pixels[pixel++];
                float red = ((val >> 16) & 0xFF);
                float green = ((val >> 8) & 0xFF);
                float blue = (val & 0xFF);
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
