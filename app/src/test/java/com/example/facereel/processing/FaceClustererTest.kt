package com.example.facereel.processing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceClustererTest {
    @Test
    fun `attaches a weak transition sample to a nearby strong temporal cluster`() {
        val groups = attachTemporalSingletons(
            groups = listOf(listOf(0), listOf(1, 2)),
            embeddings = listOf(
                floatArrayOf(1f, 0f),
                floatArrayOf(0.56f, 0.8284926f),
                floatArrayOf(0.50f, 0.8660254f),
            ),
            timestampsUs = listOf(3_250_000L, 3_500_000L, 3_750_000L),
        )

        assertEquals(1, groups.size)
        assertEquals(setOf(0, 1, 2), groups.single().toSet())
    }

    @Test
    fun `connects pose variants through strong pairwise matches`() {
        val degrees = listOf(0.0, 70.0, 35.0)
        val embeddings = degrees.map { degree ->
            val radians = Math.toRadians(degree)
            floatArrayOf(kotlin.math.cos(radians).toFloat(), kotlin.math.sin(radians).toFloat())
        }
        val groups = FaceClusterer(threshold = 0.78f).clusterIndexes(
            embeddings = embeddings,
            qualityScores = listOf(0.9, 0.8, 0.7),
        )

        assertEquals(1, groups.size)
        assertEquals(setOf(0, 1, 2), groups.single().toSet())
    }

    @Test
    fun `does not merge different identities in the ambiguous similarity band`() {
        val groups = FaceClusterer().clusterIndexes(
            embeddings = listOf(
                floatArrayOf(1f, 0f),
                floatArrayOf(0.55f, 0.83516467f),
            ),
            qualityScores = listOf(0.9, 0.8),
        )

        assertEquals(2, groups.size)
    }

    @Test
    fun `merges nearby same-person embeddings at the MobileFaceNet threshold`() {
        val groups = FaceClusterer(threshold = 0.62f).clusterIndexes(
            embeddings = listOf(
                floatArrayOf(1f, 0f),
                floatArrayOf(0.98f, 0.20f),
                floatArrayOf(0f, 1f),
            ),
            qualityScores = listOf(0.9, 0.7, 0.8),
        )

        assertEquals(2, groups.size)
        assertTrue(groups.any { it.containsAll(listOf(0, 1)) })
        assertTrue(groups.any { it == listOf(2) })
    }
}
