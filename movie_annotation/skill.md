Use the literature scan below on visual features known to be detectable with EEG. Annotate the frames of the 4 movies in `movie_annotation`. Write all the source code in the project folder.



## Literature on EEG and natural stimuli and visual processing
In terms of natural stimuli and visual processing, Electroencephalography (EEG) can encode a diverse range of features, from low-level sensory details to high-level semantic concepts. Because of its millisecond-range temporal resolution, EEG is uniquely capable of tracking the "representational timeline" of how a scene is processed in the brain—from initial physical detection to complex recognition (Orima & Motoyoshi, 2023; Wang et al., 2025).

### **1. Physical and Spatial Stimulus Features**

EEG can encode the fundamental physical properties of a visual stimulus. Recent research using naturalistic datasets (like THINGS-EEG) has shown that the brain represents these features at distinct latencies:

* **Retinal vs. Real-world Size:** EEG signals can disentangle the retinal size of an object (how much space it takes up on your eye) from its perceived real-world size (Wang et al., 2025).
* **Perceived Depth:** Spatial layout and depth information are encoded early in the processing stream, often preceding the processing of an object's actual physical size (Wang et al., 2025).
* **Low-level Visual Features:** Encoding models show that earlier stages of EEG neural processing (primarily in the occipital electrodes) are well-predicted by vision Deep Neural Networks (DNNs) that capture color, contrast, and object shape (Orima & Motoyoshi, 2023; "The time course," 2025).

### **2. Semantic and Categorical Information**

As visual processing moves from the occipital to the frontal regions, EEG encodes the "meaning" of the stimuli:

* **Object Categorization:** EEG can encode specific categories (e.g., "animal," "plant," "tool") and global properties of a scene, such as its "naturalness," "openness," or "roughness" (Orima & Motoyoshi, 2023; Wang et al., 2022).
* **Visuo-semantic Representations:** Later stages of neural processing (around 200 ms and beyond) are better predicted by Large Language Models (LLMs) than by simple vision models, suggesting that the brain is encoding the conceptual and functional meaning of the image at this stage ("The time course," 2025).
* **Abstract Concepts:** High-level semantics activate frontal-temporal networks, which can be decoded to recognize complex concepts or even generate representative images using diffusion models (Daeglau et al., 2025; Wang et al., 2022).

### **3. Dynamic and Contextual Features**

In naturalistic settings, such as watching videos or navigating environments, EEG tracks continuous changes:

* **Neural Tracking:** EEG can track the "envelope" or continuous fluctuations of a stimulus, such as the motion in a video or the congruent lip movements in a speech-heavy scene (Daeglau et al., 2025; Ronga et al., 2024).
* **Environmental Context:** EEG features like alpha and theta power differ significantly based on the environment; for example, nature scenes typically elicit lower parietal alpha power (indicating higher engagement) compared to urban scenes (Grassini, 2025).
* **Cognitive Load and Attention:** EEG encodes how much information is being "loaded" into memory during visual tasks. For instance, alpha-band power decreases as memory load increases during the viewing of naturalistic scenes (Agam & Sekuler, 2007; "Mapping neural activity," 2025).

---

### **References**

* Agam, Y., & Sekuler, R. (2007). Interactions between working memory and visual perception: An ERP/EEG study. *NeuroImage*, *36*(3), 933–942. [https://doi.org/10.1016/j.neuroimage.2007.04.014](https://www.google.com/search?q=https://doi.org/10.1016/j.neuroimage.2007.04.014)
* Daeglau, M., Otten, J., Grimm, G., Mirkovic, B., Hohmann, V., & Debener, S. (2025). Neural speech tracking in a virtual acoustic environment: audio-visual benefit for unscripted continuous speech. *arXiv*. [https://doi.org/10.48550/arxiv.2501.08124](https://www.google.com/search?q=https://doi.org/10.48550/arxiv.2501.08124)
* Grassini, S. (2025). Nature images are more visually engaging than urban images: evidence from neural oscillations in the brain. *Frontiers in Human Neuroscience*, *19*. [https://doi.org/10.3389/fnhum.2025.1575102](https://www.google.com/search?q=https://doi.org/10.3389/fnhum.2025.1575102)
* Mapping neural activity during naturalistic visual and memory search. (2025). *bioRxiv*. [https://doi.org/10.1101/2025.07.27.667084](https://www.google.com/search?q=https://doi.org/10.1101/2025.07.27.667084)
* Orima, T., & Motoyoshi, I. (2023). Spatiotemporal cortical dynamics for visual scene processing as revealed by EEG decoding. *Frontiers in Neuroscience*, *17*. [https://doi.org/10.3389/fnins.2023.1167719](https://doi.org/10.3389/fnins.2023.1167719)
* Ronga, I., et al. (2024). Brain encoding of naturalistic, continuous, and unpredictable tactile events. *Scientific Reports*. [https://pmc.ncbi.nlm.nih.gov/articles/PMC11429829/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11429829/)
* The time course of visuo-semantic representations in the human brain is captured by combining vision and language models. (2025). *eLife*. [https://elifesciences.org/reviewed-preprints/108915](https://elifesciences.org/reviewed-preprints/108915)
* Wang, Y., Wang, S., & Xu, M. (2022). Landscape perception identification and classification based on electroencephalogram (EEG) features. *International Journal of Environmental Research and Public Health*, *19*(2), 629. [https://doi.org/10.3390/ijerph19020629](https://doi.org/10.3390/ijerph19020629)
* Wang, X., et al. (2025). Human EEG and artificial neural networks reveal disentangled representations and processing timelines of object real-world size and depth in natural images. *eLife*. [https://elifesciences.org/articles/98117](https://elifesciences.org/articles/98117)

**Would you like me to focus on a specific aspect, such as the specific frequency bands (alpha, theta, gamma) associated with these visual features?**
