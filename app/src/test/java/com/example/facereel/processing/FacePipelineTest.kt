package com.example.facereel.processing

import org.junit.Assert.assertEquals
import org.junit.Test

class FacePipelineTest {
    @Test
    fun `samples eighty frames across a twenty second recording`() {
        assertEquals(80, sampleTimestamps(20_000L).size)
    }

    @Test
    fun `mouth position disambiguates upright and upside-down eye order`() {
        assertEquals(true, shouldKeepEyeOrder(0f, 0f, 2f, 0f, 1f, 1f))
        assertEquals(false, shouldKeepEyeOrder(2f, 0f, 0f, 0f, 1f, 1f))
    }

    @Test
    fun `filters tiny interface faces when a salient face is present`() {
        assertEquals(false, shouldKeepDetectedFace(faceWidth = 70, largestWidth = 300, frameWidth = 960))
        assertEquals(true, shouldKeepDetectedFace(faceWidth = 150, largestWidth = 300, frameWidth = 960))
        assertEquals(true, shouldKeepDetectedFace(faceWidth = 70, largestWidth = 70, frameWidth = 960))
    }

    @Test
    fun `normalizes reversed eye landmark angles to the nearest roll`() {
        assertEquals(-10.0, normalizedEyeRollDegrees(170.0), 0.001)
        assertEquals(10.0, normalizedEyeRollDegrees(-170.0), 0.001)
        assertEquals(25.0, normalizedEyeRollDegrees(25.0), 0.001)
    }
}
