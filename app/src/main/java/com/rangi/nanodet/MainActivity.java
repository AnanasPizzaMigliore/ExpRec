package com.rangi.nanodet;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.UseCase;
import androidx.core.app.ActivityCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.format.DateTimeParseException;
import java.time.format.ResolverStyle;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import android.os.BatteryManager;

public class MainActivity extends AppCompatActivity {
    //String Tag = "img info";
    public static int NANODET = 1;
    public static int YOLOV5S = 2;
    public static int YOLOV4_TINY = 3;

    public static int USE_MODEL = NANODET;
    public static boolean USE_GPU = false;

    public static CameraX.LensFacing CAMERA_ID = CameraX.LensFacing.BACK;

    private static final int REQUEST_CAMERA = 1;
    private static final int REQUEST_PICK_IMAGE = 2;
    private static String[] PERMISSIONS_CAMERA = {
            Manifest.permission.CAMERA
    };
    private ImageView resultImageView;
    private SeekBar nmsSeekBar;
    private SeekBar thresholdSeekBar;
    private TextView thresholdTextview;
    private TextView tvInfo;
    private double threshold = 0.3, nms_threshold = 0.7;
    private TextureView viewFinder;

    private AtomicBoolean detecting = new AtomicBoolean(false);
    private AtomicBoolean detectPhoto = new AtomicBoolean(false);
    private Handler handler = new Handler(Looper.getMainLooper());

    private long startTime = 0;
    private long endTime = 0;
    private int width;
    private int height;

    double total_fps = 0;
    int fps_count = 0;

    List<String> formats = List.of(
            "yyyy-MM-dd", "yyyy/MM/dd", "yyyy.MM.dd", "yyyy|MM|dd", "yyyyMMdd",
            "dd-MM-yyyy", "dd/MM/yyyy", "dd.MM.yyyy", "dd|MM|yyyy", "ddMMyyyy",
            "MM-dd-yyyy", "MM/dd/yyyy", "MM.dd.yyyy", "MM|dd|yyyy", "MMddyyyy",
            "MMM-dd-yyyy", "MMM/dd/yyyy", "MMM.dd.yyyy", "MMM|dd|yyyy", "MMMddyyyy",
            "dd-MMM-yyyy", "dd/MMM/yyyy", "dd.MMM.yyyy", "dd|MMM|yyyy", "ddMMMyyyy",
            "yyyy-MMM-dd", "yyyy/MMM/dd", "yyyy.MMM.dd", "yyyy|MMM|dd", "yyyyMMMdd",
            "yyyy-MM", "yyyy/MM", "yyyy.MM", "yyyy|MM", "yyyyMM",
            "MM-yyyy", "MM/yyyy", "MM.yyyy", "MM|yyyy", "MMyyyy",
            "MM-dd", "MM/dd", "MM.dd", "MM|dd", "MMdd",
            "dd-MM", "dd/MM", "dd.MM", "dd|MM", "ddMM",
            "yyyy-MMM", "yyyy/MMM", "yyyy.MMM", "yyyy|MMM", "yyyyMMM",
            "MMM-yyyy", "MMM/yyyy", "MMM.yyyy", "MMM|yyyy", "MMMMyyyy",
            "yy-MMM-dd", "yy/MMM/dd", "yy.MMM.dd", "yy|MMM|dd", "yyMMMdd",
            "dd-MMM-yy", "dd/MMM/yy", "dd.MMM.yy", "dd|MMM|yy", "ddMMMyy",
            "MMM-dd-yy", "MMM/dd/yy", "MMM.dd.yy", "MMM|dd|yy", "MMMddyy",
            "yy-MM-dd", "yy/MM/dd", "yy.MM.dd", "yy|MM|dd", "yyMMdd",
            "dd-MM-yy", "dd/MM/yy", "dd.MM.yy", "dd|MM|yy", "ddMMyy"
    );

    protected Bitmap mutableBitmap;

    ExecutorService detectService = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        BatteryManager batteryManager = (BatteryManager) getSystemService(BATTERY_SERVICE);
        int chargeCounter = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                int chargeCounter = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
                saveToFile(chargeCounter);
                handler.postDelayed(this, 60000); // 1 minute (60000 ms)
            }
        }, 60000); // 1 minute (60000 ms)

        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    this,
                    PERMISSIONS_CAMERA,
                    REQUEST_CAMERA
            );
            finish();
        }


        if (USE_MODEL == YOLOV5S) {
            YOLOv5.init(getAssets(), USE_GPU);
        } else if (USE_MODEL == YOLOV4_TINY) {
            YOLOv4.init(getAssets(), USE_GPU);
        }  else if (USE_MODEL == NANODET) {
            NanoDet.init(getAssets(), USE_GPU);
            CRNN.init(getAssets(), USE_GPU);
        }
        resultImageView = findViewById(R.id.imageView);
        thresholdTextview = findViewById(R.id.valTxtView);
        tvInfo = findViewById(R.id.tv_info);
        nmsSeekBar = findViewById(R.id.nms_seek);
        thresholdSeekBar = findViewById(R.id.threshold_seek);
        if (USE_MODEL != YOLOV5S && USE_MODEL != NANODET) {
            nmsSeekBar.setEnabled(false);
            thresholdSeekBar.setEnabled(false);
        } else if (USE_MODEL == YOLOV5S) {
            threshold = 0.3f;
            nms_threshold = 0.7f;
        } else {
            threshold = 0.4f;
            nms_threshold = 0.6f;
        }
        nmsSeekBar.setProgress((int) (nms_threshold * 100));
        thresholdSeekBar.setProgress((int) (threshold * 100));
        final String format = "Thresh: %.2f, NMS: %.2f";
        thresholdTextview.setText(String.format(Locale.ENGLISH, format, threshold, nms_threshold));
        nmsSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                nms_threshold = i / 100.f;
                thresholdTextview.setText(String.format(Locale.ENGLISH, format, threshold, nms_threshold));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                threshold = i / 100.f;
                thresholdTextview.setText(String.format(Locale.ENGLISH, format, threshold, nms_threshold));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        Button inference = findViewById(R.id.button);
        inference.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent, REQUEST_PICK_IMAGE);
            }
        });

        resultImageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                detectPhoto.set(false);
            }
        });

        viewFinder = findViewById(R.id.view_finder);
        viewFinder.addOnLayoutChangeListener(new View.OnLayoutChangeListener() {
            @Override
            public void onLayoutChange(View view, int i, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
                updateTransform();
            }
        });

        viewFinder.post(new Runnable() {
            @Override
            public void run() {
                startCamera();
            }
        });
    }



    private void updateTransform() {
        Matrix matrix = new Matrix();
        // Compute the center of the view finder
        float centerX = viewFinder.getWidth() / 2f;
        float centerY = viewFinder.getHeight() / 2f;

        float[] rotations = {0, 90, 180, 270};
        // Correct preview output to account for display rotation
        float rotationDegrees = rotations[viewFinder.getDisplay().getRotation()];

        matrix.postRotate(-rotationDegrees, centerX, centerY);

        // Finally, apply transformations to our TextureView
        viewFinder.setTransform(matrix);
    }

    private void startCamera() {
        CameraX.unbindAll();
        // 1. preview
        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(CAMERA_ID)
//                .setTargetAspectRatio()  // 宽高比
                .setTargetResolution(new Size(320, 320))  // 分辨率
                .build();

        Preview preview = new Preview(previewConfig);
        preview.setOnPreviewOutputUpdateListener(new Preview.OnPreviewOutputUpdateListener() {
            @Override
            public void onUpdated(Preview.PreviewOutput output) {
                ViewGroup parent = (ViewGroup) viewFinder.getParent();
                parent.removeView(viewFinder);
                parent.addView(viewFinder, 0);

                viewFinder.setSurfaceTexture(output.getSurfaceTexture());
                updateTransform();
            }
        });
        DetectAnalyzer detectAnalyzer = new DetectAnalyzer();
        CameraX.bindToLifecycle((LifecycleOwner) this, preview, gainAnalyzer(detectAnalyzer));

    }


    private UseCase gainAnalyzer(DetectAnalyzer detectAnalyzer) {
        ImageAnalysisConfig.Builder analysisConfigBuilder = new ImageAnalysisConfig.Builder();
        analysisConfigBuilder.setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE);
        analysisConfigBuilder.setTargetResolution(new Size(320, 320));  // 输出预览图像尺寸
        ImageAnalysisConfig config = analysisConfigBuilder.build();
        ImageAnalysis analysis = new ImageAnalysis(config);
        analysis.setAnalyzer(detectAnalyzer);
        return analysis;
    }

    private Bitmap imageToBitmap(ImageProxy image) {
        byte[] nv21 = imagetToNV21(image);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private byte[] imagetToNV21(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ImageProxy.PlaneProxy y = planes[0];
        ImageProxy.PlaneProxy u = planes[1];
        ImageProxy.PlaneProxy v = planes[2];
        ByteBuffer yBuffer = y.getBuffer();
        ByteBuffer uBuffer = u.getBuffer();
        ByteBuffer vBuffer = v.getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] nv21 = new byte[ySize + uSize + vSize];
        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        return nv21;
    }

    private class DetectAnalyzer implements ImageAnalysis.Analyzer {

        @Override
        public void analyze(ImageProxy image, final int rotationDegrees) {
            detectOnModel(image, rotationDegrees);
        }
    }


    private void detectOnModel(ImageProxy image, final int rotationDegrees) {
        if (detecting.get() || detectPhoto.get()) {
            return;
        }
        detecting.set(true);
        startTime = System.currentTimeMillis();
        final Bitmap bitmapsrc = imageToBitmap(image);  // 格式转换
        if (detectService == null) {
            detecting.set(false);
            return;
        }
        detectService.execute(new Runnable() {
            @Override
            public void run() {
                Matrix matrix = new Matrix();
                matrix.postRotate(rotationDegrees);
                width = bitmapsrc.getWidth();
                height = bitmapsrc.getHeight();
                Bitmap bitmap = Bitmap.createBitmap(bitmapsrc, 0, 0, Math.min(width, height), Math.min(width, height), matrix, false);

                Box[] result = null;
                if (USE_MODEL == YOLOV5S) {
                    result = YOLOv5.detect(bitmap, threshold, nms_threshold);
                } else if (USE_MODEL == YOLOV4_TINY) {
                    result = YOLOv4.detect(bitmap, threshold, nms_threshold);
                } else if (USE_MODEL == NANODET) {
                    result = NanoDet.detect(bitmap, threshold, nms_threshold);
                }
                if (result == null) {
                    detecting.set(false);
                    return;
                }
                mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                if (USE_MODEL == YOLOV5S || USE_MODEL == YOLOV4_TINY || USE_MODEL == NANODET) {
                    mutableBitmap = drawBoxRects(mutableBitmap, result);
                }
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        detecting.set(false);
                        if (detectPhoto.get()) {
                            return;
                        }
                        resultImageView.setImageBitmap(mutableBitmap);
                        endTime = System.currentTimeMillis();
                        long dur = endTime - startTime;
                        float fps = (float) (1000.0 / dur);
                        total_fps = (total_fps == 0) ? fps : (total_fps + fps);
                        fps_count++;
                        String modelName = getModelName();

                        tvInfo.setText(String.format(Locale.CHINESE,
                                "%s\nSize: %dx%d\nTime: %.3f s\nFPS: %.3f\nAVG_FPS: %.3f",
                                modelName, height, width, dur / 1000.0, fps, (float) total_fps / fps_count));
                    }
                });
            }
        });
    }

    public void saveToFile(int chargeCounter) {
        String fileName = "battery_info_int8_gpu.txt";
        String timeStamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(new Date());
        String content = "Timestamp: " + timeStamp + " Charge Counter: " + chargeCounter + " μAh\n";
        FileOutputStream fos = null;
        //Log.d(" Charge Counter: " , String.valueOf(chargeCounter));
        try {
            fos = openFileOutput(fileName, Context.MODE_PRIVATE | Context.MODE_APPEND);
            fos.write(content.getBytes());
            Toast.makeText(this, "Battery info saved", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    protected Bitmap drawBoxRects(Bitmap mutableBitmap, Box[] results) {
        if (results == null || results.length == 0) {
            return mutableBitmap;
        }
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint boxPaint = new Paint();
        boxPaint.setAlpha(200);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4 * mutableBitmap.getWidth() / 800.0f);
        boxPaint.setTextSize(40 * mutableBitmap.getWidth() / 800.0f);
        for (Box box : results) {
            boxPaint.setColor(box.getColor());
            boxPaint.setStyle(Paint.Style.FILL);
            Bitmap img;
            String text;
            try {
                int x = (int) (box.x0 - 5);
                int y = (int) (box.y0 - 5);
                int width = (int) (box.x1 - box.x0 + 10);
                int height = (int) (box.y1 - box.y0 + 10);

                // Check boundaries
                if (x < 0 || y < 0 || x + width > mutableBitmap.getWidth() || y + height > mutableBitmap.getHeight()) {
                    break;
                } else {
                    img = Bitmap.createBitmap(mutableBitmap, x, y, width, height);
                    text = reg(img);
                }
                canvas.drawText(text, box.x0 + 3, box.y0 + 100 * mutableBitmap.getWidth() / 1000.0f, boxPaint);
                boxPaint.setStyle(Paint.Style.STROKE);
                canvas.drawRect(box.getRect(), boxPaint);
            } catch (IllegalArgumentException e) {
                e.printStackTrace();
                Log.e("BitmapCreation", "Failed to create bitmap: " + e.getMessage());
            }
        }
        return mutableBitmap;
    }

    public static LocalDate parseDates(String dateStr, List<String> formats) {
        LocalDate currentDate = LocalDate.now();
        List<LocalDate> parsedDates = new ArrayList<>();

        // Attempt to parse the date string with each format
        for (String format : formats) {
            try {
                //DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format, Locale.ENGLISH);
                var formatter = new DateTimeFormatterBuilder()
                        .parseCaseInsensitive()
                        .appendPattern(format)
                        .parseDefaulting(ChronoField.DAY_OF_MONTH, 1)
                        .parseDefaulting(ChronoField.YEAR_OF_ERA, LocalDate.now().getYear())
                        .toFormatter(Locale.ENGLISH);
                parsedDates.add(LocalDate.parse(dateStr, formatter));
            } catch (DateTimeParseException e) {
                // Ignore invalid formats
            }
        }

        // Find the closest date to the current date
        return parsedDates.stream()
                .min(Comparator.comparingLong(date -> Math.abs(ChronoUnit.DAYS.between(currentDate, date))))
                .orElseThrow(() -> new IllegalArgumentException("No valid date found for the given formats"));
    }

    public static DateTimeFormatter createFormatter(List<String> patterns, Locale locale) {
        DateTimeFormatterBuilder builder = new DateTimeFormatterBuilder()
                .parseCaseInsensitive(); // Enable case-insensitive parsing

        for (String pattern : patterns) {
            builder.appendOptional(DateTimeFormatter.ofPattern(pattern));
        }

        return builder.toFormatter(locale).withResolverStyle(ResolverStyle.STRICT);
    }

    protected String reg(Bitmap img){
        if (img == null) {
            return null;
        }
        else {
            var text = CRNN.recognize(img);
            try {
                LocalDate closestDate = parseDates(text, formats);
                return closestDate.toString();
            } catch (IllegalArgumentException e) {
                return text;
            }
        }
    }


    protected String getModelName() {
        String modelName = "ohhhhh";
        if (USE_MODEL == YOLOV5S) {
            modelName = "YOLOv5s";
        } else if (USE_MODEL == YOLOV4_TINY) {
            modelName = "YOLOv4-tiny";
        } else if (USE_MODEL == NANODET) {
            modelName = "NanoDet";
        }
        return USE_GPU ? "GPU: " + modelName : "CPU: " + modelName;
    }

    @Override
    protected void onDestroy() {
        if (detectService != null) {
            detectService.shutdown();
            detectService = null;
        }
        CameraX.unbindAll();
        super.onDestroy();
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera Permission!", Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data == null) {
            return;
        }
        detectPhoto.set(true);
        Bitmap image = getPicture(data.getData());
        if (image == null) {
            Toast.makeText(this, "Photo is null", Toast.LENGTH_SHORT).show();
            return;
        }
        Bitmap mutableBitmap = image.copy(Bitmap.Config.ARGB_8888, true);

        Box[] result = null;
        if (USE_MODEL == YOLOV5S) {
            result = YOLOv5.detect(image, threshold, nms_threshold);
        } else if (USE_MODEL == YOLOV4_TINY) {
            result = YOLOv4.detect(image, threshold, nms_threshold);
        } else if (USE_MODEL == NANODET) {
            result = NanoDet.detect(image, threshold, nms_threshold);
        }
        if (USE_MODEL == YOLOV5S || USE_MODEL == YOLOV4_TINY|| USE_MODEL == NANODET) {
            mutableBitmap = drawBoxRects(mutableBitmap, result);
        }
        resultImageView.setImageBitmap(mutableBitmap);
    }

    public Bitmap getPicture(Uri selectedImage) {
        String[] filePathColumn = {MediaStore.Images.Media.DATA};
        Cursor cursor = this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
        if (cursor == null) {
            return null;
        }
        cursor.moveToFirst();
        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
        String picturePath = cursor.getString(columnIndex);
        cursor.close();
        Bitmap bitmap = BitmapFactory.decodeFile(picturePath);
        if (bitmap == null) {
            return null;
        }
        int rotate = readPictureDegree(picturePath);
        return rotateBitmapByDegree(bitmap, rotate);
    }

    public int readPictureDegree(String path) {
        int degree = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            degree = switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90 -> 90;
                case ExifInterface.ORIENTATION_ROTATE_180 -> 180;
                case ExifInterface.ORIENTATION_ROTATE_270 -> 270;
                default -> degree;
            };
        } catch (IOException e) {
            e.printStackTrace();
        }
        return degree;
    }

    public Bitmap rotateBitmapByDegree(Bitmap bm, int degree) {
        Bitmap returnBm = null;
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);
        try {
            returnBm = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(),
                    bm.getHeight(), matrix, true);
        } catch (OutOfMemoryError e) {
            e.printStackTrace();
        }
        if (returnBm == null) {
            returnBm = bm;
        }
        if (bm != returnBm) {
            bm.recycle();
        }
        return returnBm;
    }


}
