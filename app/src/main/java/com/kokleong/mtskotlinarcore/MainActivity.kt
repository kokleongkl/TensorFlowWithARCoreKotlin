package com.kokleong.mtskotlinarcore

import androidx.appcompat.app.AppCompatActivity


import com.google.ar.core.Config
import com.google.ar.core.Plane

import com.google.ar.core.TrackingState


import android.content.Context
import android.content.ContextWrapper
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.sceneform.FrameTime
import com.google.ar.sceneform.Scene
import com.google.ar.sceneform.ux.ArFragment
import com.kokleong.mtskotlinarcore.tensorflow.lite.IClassifier
import com.kokleong.mtskotlinarcore.tensorflow.lite.TensorFlowClassifier
import com.kokleong.mtskotlinarcore.R


import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    //ar required variables
    private var arFragment: ArFragment? = null
    private var shouldTakePhoto = true
    private var classifier: IClassifier? = null
    private val executor = Executors.newSingleThreadExecutor()

    //UI variables
    private var sheetBehavior: BottomSheetBehavior<View>? = null
    private var bottomSheet: LinearLayout? = null
    private var bottomButton: Button? = null
    private var mMedicineName: TextView? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initTensorFlowAndLoadModel()
        bottomButton = findViewById<Button>(R.id.bottom_btn)
        bottomSheet = findViewById<LinearLayout>(R.id.bottom_sheet)
        mMedicineName = findViewById<TextView>(R.id.medicine_name)
        sheetBehavior = BottomSheetBehavior.from(bottomSheet)

        bottomButton!!.setOnClickListener {
            bottomSheet!!.visibility = View.GONE
            shouldTakePhoto = true
        }

        sheetBehavior!!.setBottomSheetCallback(object : BottomSheetBehavior.BottomSheetCallback() {
            override fun onStateChanged(view: View, newState: Int) {
                when (newState) {
                    BottomSheetBehavior.STATE_HIDDEN -> {
                    }
                    BottomSheetBehavior.STATE_EXPANDED -> {
                        bottomButton!!.text = "Close"
                    }
                    BottomSheetBehavior.STATE_COLLAPSED -> {
                        bottomButton!!.text = "Expand"
                    }
                    BottomSheetBehavior.STATE_DRAGGING -> {
                    }
                    BottomSheetBehavior.STATE_SETTLING -> {
                    }
                }
            }

           override fun onSlide(view: View, v: Float) {

            }
        })


        // render = new RenderHandler(MainActivity.this);
        arFragment = supportFragmentManager.findFragmentById(R.id.sceneform_fragment) as ArFragment?
        arFragment!!.arSceneView.scene.addOnUpdateListener(Scene.OnUpdateListener { this.onUpdateFrame(it) })

    }


    private fun onUpdateFrame(frameTime: FrameTime) {
        val session = arFragment!!.arSceneView.session
        val config = Config(session!!)
        config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
        config.focusMode = Config.FocusMode.AUTO
        session.configure(config)
        val frame = arFragment!!.arSceneView.arFrame ?: return

        //if there is no frame don't process anything
        // If Arcore is not tracking yet then don't process anything
        if (frame.camera.trackingState != TrackingState.TRACKING) {
            return
        }
        //if ARCore is tracking get start processing
        if (frame.camera.trackingState == TrackingState.TRACKING) {
            if (shouldTakePhoto) {

                try {
                    //take photo convert photo to Bitmap format so that it could be use in the TensorflowImageClassifierClass for detection
                    val img = frame.acquireCameraImage()
                    val nv21: ByteArray
                    val cw = ContextWrapper(applicationContext)
                    val fileName = "test.jpg"
                    val dir = cw.getDir("imageDir", Context.MODE_PRIVATE)
                    val file = File(dir, fileName)
                    val outputStream: FileOutputStream
                    try {

                        outputStream = FileOutputStream(file)
                        val yBuffer = img.planes[0].buffer
                        val uBuffer = img.planes[1].buffer
                        val vBuffer = img.planes[2].buffer

                        val ySize = yBuffer.remaining()
                        val uSize = uBuffer.remaining()
                        val vSize = vBuffer.remaining()

                        nv21 = ByteArray(ySize + uSize + vSize)

                        yBuffer.get(nv21, 0, ySize)
                        vBuffer.get(nv21, ySize, vSize)
                        uBuffer.get(nv21, ySize + vSize, uSize)

                        val width = img.width
                        val height = img.height

                        img.close()


                        val out = ByteArrayOutputStream()
                        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
                        yuv.compressToJpeg(Rect(0, 0, width, height), 100, out)
                        val byteArray = out.toByteArray()

                        val bitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)
                        val matrix = Matrix()
                        matrix.postRotate(90f)
                        if (bitmap != null) {
                            Log.i("bitmap ", "contains data")

                        }
                        val portraitBitmap =
                            Bitmap.createBitmap(bitmap!!, 0, 0, bitmap.width, bitmap.height, matrix, true)


                        val scaledBitmap = Bitmap.createScaledBitmap(portraitBitmap, INPUT_SIZE, INPUT_SIZE, false)
                        scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                        outputStream.flush()
                        outputStream.close()

                        val results = classifier!!.recognizeImage(scaledBitmap)
                        //change according with your model
                        if (results.get(0).confidence > 0.995) {
                            Log.i("results", results.toString())
                            for (plane in frame.getUpdatedTrackables(Plane::class.java)) {
                                if (results.get(0).title.equals("Anarex")) {
                                    shouldTakePhoto = false
                                    mMedicineName!!.setText(results.get(0).title)
                                    bottomSheet!!.visibility = View.VISIBLE
                                    //Anchor anchor = arFragment.getArSceneView().getSession().createAnchor(plane.getCenterPose());
                                    //render.placeObject(anchor, arFragment, results.get(0).getTitle());
                                } else if (results.get(0).title.equals("Charcoal")) {
                                    shouldTakePhoto = false
                                    mMedicineName!!.setText(results.get(0).title)
                                    //sheetBehavior.setState(BottomSheetBehavior.STATE_EXPANDED);
                                    bottomSheet!!.visibility = View.VISIBLE
                                    //Anchor anchor = arFragment.getArSceneView().getSession().createAnchor(plane.getCenterPose());
                                    //render.placeObject(anchor, arFragment, results.get(0).getTitle());

                                } else if (results.get(0).title.equals("Dhamotil")) {
                                    shouldTakePhoto = false
                                    mMedicineName!!.setText(results.get(0).title)
                                    // sheetBehavior.setState(BottomSheetBehavior.STATE_EXPANDED);
                                    bottomSheet!!.visibility = View.VISIBLE
                                    //Anchor anchor = arFragment.getArSceneView().getSession().createAnchor(plane.getCenterPose());
                                    //render.placeObject(anchor, arFragment, results.get(0).getTitle());

                                } else if (results.get(0).title.equals("Fucon")) {
                                    shouldTakePhoto = false
                                    mMedicineName!!.setText(results.get(0).title)
                                    //sheetBehavior.setState(BottomSheetBehavior.STATE_EXPANDED);
                                    bottomSheet!!.visibility = View.VISIBLE
                                    // Anchor anchor = arFragment.getArSceneView().getSession().createAnchor(plane.getCenterPose());
                                    // render.placeObject(anchor, arFragment, results.get(0).getTitle());
                                }
                            }
                        }

                    } catch (e: Exception) {
                        e.printStackTrace()
                    }

                } catch (e: NotYetAvailableException) {
                    e.printStackTrace()
                }

            }
        }

    }

    private fun initTensorFlowAndLoadModel() {
        executor.execute {
            try {
                classifier = TensorFlowClassifier.create(
                    assets,
                    MODEL_PATH,
                    LABEL_PATH,
                    INPUT_SIZE,
                    QUANT
                )
            } catch (e: IOException) {
                e.printStackTrace()
                throw RuntimeException("Error initializing TensorFlow!", e)
            }
        }
    }

    companion object {
        //    private RenderHandler render;
        //Tensorflow required variables
        private val MODEL_PATH = "medicine_quant.tflite"
        private val QUANT = false
        private val LABEL_PATH = "labels.txt"
        private val INPUT_SIZE = 128
    }


}
