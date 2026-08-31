package com.example.facereel.processing

import kotlin.math.sqrt

/** Explainable centroid clustering for L2-normalized MobileFaceNet embeddings. */
class FaceClusterer(private val threshold: Float = 0.75f) {
    fun cluster(candidates: List<FaceCandidate>): List<FaceCluster> {
        val groups = clusterIndexes(
            candidates.map { it.embedding },
            candidates.map { it.qualityScore },
        )
        return attachTemporalSingletons(
            groups,
            candidates.map { it.embedding },
            candidates.map { it.timestampUs },
        ).map { indexes ->
            val members = indexes.map { candidates[it] }.toMutableList()
            FaceCluster(members, normalizedCandidateCentroid(members))
        }
    }

    /** Kept Android-free so threshold behavior has a fast JVM regression test. */
    internal fun clusterIndexes(embeddings: List<FloatArray>, qualityScores: List<Double>): List<List<Int>> {
        require(embeddings.size == qualityScores.size)
        val visited = BooleanArray(embeddings.size)
        val clusters = mutableListOf<List<Int>>()
        qualityScores.indices.sortedByDescending { qualityScores[it] }.forEach { seed ->
            if (visited[seed]) return@forEach
            val component = mutableListOf<Int>()
            val pending = ArrayDeque<Int>().apply { add(seed) }
            visited[seed] = true
            while (pending.isNotEmpty()) {
                val current = pending.removeFirst()
                component += current
                embeddings.indices.forEach { candidate ->
                    if (!visited[candidate] && cosine(embeddings[current], embeddings[candidate]) >= threshold) {
                        visited[candidate] = true
                        pending.add(candidate)
                    }
                }
            }
            clusters += component
        }
        return clusters
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float = a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()

    private fun normalizedCandidateCentroid(members: List<FaceCandidate>): FloatArray {
        return normalizedEmbeddingCentroid(members.map { it.embedding })
    }

    private fun normalizedEmbeddingCentroid(embeddings: List<FloatArray>): FloatArray {
        val output = FloatArray(embeddings.first().size)
        embeddings.forEach { embedding -> output.indices.forEach { output[it] += embedding[it] } }
        val magnitude = sqrt(output.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-6f)
        output.indices.forEach { output[it] /= magnitude }
        return output
    }
}

/**
 * A brief blur/occlusion can fall below the identity edge threshold. Attach only
 * singleton border samples to an already-established cluster when they are both
 * temporally adjacent and moderately similar; they can never bridge two clusters.
 */
internal fun attachTemporalSingletons(
    groups: List<List<Int>>,
    embeddings: List<FloatArray>,
    timestampsUs: List<Long>,
    attachmentThreshold: Float = 0.52f,
    maxGapUs: Long = 500_000L,
): List<List<Int>> {
    val result = groups.map { it.toMutableList() }.toMutableList()
    result.filter { it.size == 1 }.toList().forEach { singleton ->
        if (singleton !in result) return@forEach
        val sample = singleton.single()
        val target = result.asSequence()
            .filter { it.size >= 2 }
            .map { cluster ->
                val similarity = cluster.maxOf { cosineOf(embeddings[sample], embeddings[it]) }
                val gap = cluster.minOf { kotlin.math.abs(timestampsUs[sample] - timestampsUs[it]) }
                Triple(cluster, similarity, gap)
            }
            .filter { (_, similarity, gap) -> similarity >= attachmentThreshold && gap <= maxGapUs }
            .maxByOrNull { (_, similarity, _) -> similarity }
            ?.first
        if (target != null) {
            target += sample
            result.remove(singleton)
        }
    }
    return result
}

private fun cosineOf(a: FloatArray, b: FloatArray): Float =
    a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()
