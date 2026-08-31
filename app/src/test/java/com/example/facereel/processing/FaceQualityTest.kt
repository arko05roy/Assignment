package com.example.facereel.processing

import org.junit.Assert.assertTrue
import org.junit.Test

class FaceQualityTest {
    @Test
    fun `sharp smaller crop outranks a large motion-blurred crop`() {
        val sharp = qualityRank(area = 0.03, sharpness = 60.0, yaw = 0f, roll = 0f)
        val blurred = qualityRank(area = 0.35, sharpness = 20.0, yaw = 0f, roll = 0f)

        assertTrue(sharp > blurred)
    }
}
