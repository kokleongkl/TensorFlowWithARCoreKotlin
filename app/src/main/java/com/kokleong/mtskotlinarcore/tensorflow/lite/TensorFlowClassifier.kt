package com.kokleong.mtskotlinarcore.tensorflow.lite

import android.annotation.SuppressLint
import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.*
import java.lang.Float
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.ArrayList
import kotlin.experimental.and

class TensorFlowClassifier(
    var interpreter : Interpreter? = null,
    var inputSize : Int = 0,
    var labelList : List<String> =emptyList(),
    var quant: Boolean = false

): IClassifier{
    companion object{
        private val MAX_RESULT = 3;
        private val BATCH_SIZE = 1;
        private val PIXEL_SIZE = 3;
        private val THRESHOLD = 0.1f;

        private val IMAGE_MEAN = 0;
        private val IMAGE_STD = 255.0f;

        @Throws(IOException::class)
        fun create(
            assetManager :  AssetManager,
            modelPath : String,
            labelPath : String,
            inputSize: Int,
            quant : Boolean
        ):IClassifier{
            val classifier = TensorFlowClassifier()
            classifier.interpreter =  Interpreter(classifier.loadModelFile(assetManager,modelPath));
            classifier.labelList =  classifier.loadLabelList(assetManager,labelPath);
            classifier.inputSize = inputSize;
            classifier.quant = quant;
            return classifier;

        }
    }

    override fun recognizeImage(bitmap: Bitmap): List<IClassifier.Recognition> {
        val byteBuffer = convertBitMapToByteBuffer(bitmap);
        if(quant){
            val result = Array(1){ByteArray(labelList.size)};
            interpreter!!.run(byteBuffer,result);
            return getSortedResultByte(result)

        }else{
            val result = Array(1){FloatArray(labelList.size)};
            interpreter!!.run(byteBuffer,result);
            return getSortedResultFloat(result);

        }


    }
    override fun close(){
        interpreter!!.close();
        interpreter = null;
    }
    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager,modelPath: String):MappedByteBuffer{
        val fileDescriptor = assetManager.openFd(modelPath);
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor);
        val fileChannel =  inputStream.channel;
        val startOffset = fileDescriptor.startOffset;
        val declaredLength = fileDescriptor.declaredLength;
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    @Throws(IOException::class)
    private fun loadLabelList(assetManager: AssetManager,labelPath: String):List<String>{
        val labelList = ArrayList<String>();
        val reader = BufferedReader(InputStreamReader(assetManager.open(labelPath)));
        while (true){
            val line = reader.readLine() ?: break
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }


    private fun convertBitMapToByteBuffer(bitmap: Bitmap) : ByteBuffer{
        val byteBuffer :ByteBuffer;
        if(quant){
            byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        }
        else{
            byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        val intValue =  IntArray(inputSize * inputSize);
        bitmap.getPixels(intValue,0,bitmap.width,0,0,bitmap.width,bitmap.height);
        var pixel = 0;
        for(i in 0 until inputSize){
            for(j in 0 until inputSize){
                val `val` = intValue[pixel++]
                if(quant) {

                    byteBuffer.put((`val` shr 16 and 0xFF).toByte());
                    byteBuffer.put((`val` shr 8 and 0xFF).toByte());
                    byteBuffer.put((`val` and 0xFF).toByte())
                }else{
                    byteBuffer.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    byteBuffer.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    byteBuffer.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }
        return byteBuffer;

    }
    @SuppressLint("DefaultLocale")
    private fun getSortedResultByte(labelProbArray: Array<ByteArray>): List<IClassifier.Recognition> {

        val pq = PriorityQueue<IClassifier.Recognition>(
            MAX_RESULT,
            Comparator<IClassifier.Recognition>  { (_, _, confidence1), (_, _, confidence2) -> Float.compare(confidence1, confidence2) })

        for (i in labelList.indices) {
            val confidence = (labelProbArray[0][i].toInt() and  0xff) / 255.0f
            if (confidence > THRESHOLD) {
                pq.add(
                    IClassifier.Recognition(
                        "" + i,
                        if (labelList.size > i) labelList[i] else "unknown",
                        confidence, quant
                    )
                )
            }
        }

        val recognitions = java.util.ArrayList<IClassifier.Recognition>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULT)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }

        return recognitions
    }

    @SuppressLint("DefaultLocale")
    private fun getSortedResultFloat(labelProbArray: Array<FloatArray>): List<IClassifier.Recognition> {

        val pq = PriorityQueue<IClassifier.Recognition>(
            MAX_RESULT,
            Comparator<IClassifier.Recognition>  { (_, _, confidence1), (_, _, confidence2) -> Float.compare(confidence1, confidence2) })

        for (i in labelList.indices) {
            val confidence = labelProbArray[0][i]
            if (confidence > THRESHOLD) {
                pq.add(
                    IClassifier.Recognition(
                        "" + i,
                        if (labelList.size > i) labelList[i] else "unknown",
                        confidence, quant
                    )
                )
            }
        }

        val recognitions = java.util.ArrayList<IClassifier.Recognition>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULT)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }

        return recognitions
    }


}