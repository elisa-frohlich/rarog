#ifndef RAROG_INCLUDE_TRANSFORM_PIPELINE_H
#define RAROG_INCLUDE_TRANSFORM_PIPELINE_H

namespace rarog {

void registerNasbenchLoweringPipeline();
void registerInstrumentMallocPipeline();
void registerReorderFreesPipeline();

} // namespace rarog

#endif // RAROG_INCLUDE_TRANSFORM_PIPELINE_H