// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#include "ebsynth.h"
#include "ebsynth_cpu.h"
#include "ebsynth_cuda.h"

#include <cmath>
#include <cstdio>

EBSYNTH_API
void ebsynthRun(int    ebsynthBackend,
                int    numStyleChannels,
                int    numGuideChannels,
                int    sourceWidth,
                int    sourceHeight,
                void*  sourceStyleData,
                void*  sourceGuideData,
                int    targetWidth,
                int    targetHeight,
                void*  targetGuideData,
                void*  targetModulationData,
                float* styleWeights,
                float* guideWeights,
                float  uniformityWeight,
                int    patchSize,
                int    voteMode,
                int    numPyramidLevels,
                int*   numSearchVoteItersPerLevel,
                int*   numPatchMatchItersPerLevel,
                int*   stopThresholdPerLevel,
                int    extraPass3x3,
                void*  outputNnfData,
                void*  outputImageData)
{
    void (*backendDispatch)(int, int, int, int, void*, void*, int, int, void*, void*, float*, float*, float, int, int, int, int*, int*, int*, int, void*, void*) = 0;

    if (ebsynthBackend == EBSYNTH_BACKEND_CPU) {
        backendDispatch = ebsynthRunCpu;
    } else if (ebsynthBackend == EBSYNTH_BACKEND_CUDA) {
        backendDispatch = ebsynthRunCuda;
    } else if (ebsynthBackend == EBSYNTH_BACKEND_AUTO) {
        backendDispatch = ebsynthBackendAvailableCuda() ? ebsynthRunCuda : ebsynthRunCpu;
    }

    if (backendDispatch != 0) {
        backendDispatch(numStyleChannels,
                        numGuideChannels,
                        sourceWidth,
                        sourceHeight,
                        sourceStyleData,
                        sourceGuideData,
                        targetWidth,
                        targetHeight,
                        targetGuideData,
                        targetModulationData,
                        styleWeights,
                        guideWeights,
                        uniformityWeight,
                        patchSize,
                        voteMode,
                        numPyramidLevels,
                        numSearchVoteItersPerLevel,
                        numPatchMatchItersPerLevel,
                        stopThresholdPerLevel,
                        extraPass3x3,
                        outputNnfData,
                        outputImageData);
    }
}

EBSYNTH_API
int ebsynthBackendAvailable(int ebsynthBackend)
{
    if (ebsynthBackend == EBSYNTH_BACKEND_CPU) {
        return ebsynthBackendAvailableCpu();
    } else if (ebsynthBackend == EBSYNTH_BACKEND_CUDA) {
        return ebsynthBackendAvailableCuda();
    } else if (ebsynthBackend == EBSYNTH_BACKEND_AUTO) {
        return ebsynthBackendAvailableCpu() || ebsynthBackendAvailableCuda();
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "jzq.h"

template <typename FUNC>
bool tryToParseArg(const std::vector<std::string>& args, int* inout_argi, const char* name, bool* out_fail, FUNC handler)
{
    int&  argi = *inout_argi;
    bool& fail = *out_fail;

    if (argi < 0 || argi >= args.size()) {
        fail = true;
        return false;
    }

    if (args[argi] == name) {
        argi++;
        fail = !handler();
        return true;
    }

    fail = false;
    return false;
}

bool tryToParseIntArg(const std::vector<std::string>& args, int* inout_argi, const char* name, int* out_value, bool* out_fail)
{
    return tryToParseArg(args, inout_argi, name, out_fail, [&] {
        int& argi = *inout_argi;
        if (argi < args.size()) {
            const std::string& arg = args[argi];
            try {
                std::size_t pos = 0;
                *out_value      = std::stoi(arg, &pos);
                if (pos != arg.size()) {
                    printf("error: bad %s argument '%s'\n", name, arg.c_str());
                    return false;
                }
                return true;
            } catch (...) {
                printf("error: bad %s argument '%s'\n", name, arg.c_str());
                return false;
            }
        }
        printf("error: missing argument for the %s option\n", name);
        return false;
    });
}

bool tryToParseFloatArg(const std::vector<std::string>& args, int* inout_argi, const char* name, float* out_value, bool* out_fail)
{
    return tryToParseArg(args, inout_argi, name, out_fail, [&] {
        int& argi = *inout_argi;
        if (argi < args.size()) {
            const std::string& arg = args[argi];
            try {
                std::size_t pos = 0;
                *out_value      = std::stof(arg, &pos);
                if (pos != arg.size()) {
                    printf("error: bad %s argument '%s'\n", name, arg.c_str());
                    return false;
                }
                return true;
            } catch (...) {
                printf("error: bad %s argument '%s'\n", name, args[argi].c_str());
                return false;
            }
        }
        printf("error: missing argument for the %s option\n", name);
        return false;
    });
}

bool tryToParseStringArg(const std::vector<std::string>& args, int* inout_argi, const char* name, std::string* out_value, bool* out_fail)
{
    return tryToParseArg(args, inout_argi, name, out_fail, [&] {
        int& argi = *inout_argi;
        if (argi < args.size()) {
            *out_value = args[argi];
            return true;
        }
        printf("error: missing argument for the %s option\n", name);
        return false;
    });
}

bool tryToParseStringPairArg(const std::vector<std::string>& args, int* inout_argi, const char* name, std::pair<std::string, std::string>* out_value, bool* out_fail)
{
    return tryToParseArg(args, inout_argi, name, out_fail, [&] {
        int& argi = *inout_argi;
        if ((argi + 1) < args.size()) {
            *out_value = std::make_pair(args[argi], args[argi + 1]);
            argi++;
            return true;
        }
        printf("error: missing argument for the %s option\n", name);
        return false;
    });
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

unsigned char* tryLoad(const std::string& fileName, int* width, int* height)
{
    unsigned char* data = stbi_load(fileName.c_str(), width, height, NULL, 4);
    if (data == NULL) {
        printf("error: failed to load '%s'\n", fileName.c_str());
        printf("%s\n", stbi_failure_reason());
        exit(1);
    }
    return data;
}

int evalNumChannels(const unsigned char* data, const int numPixels)
{
    bool isGray   = true;
    bool hasAlpha = false;

    for (int xy = 0; xy < numPixels; xy++) {
        const unsigned char r = data[xy * 4 + 0];
        const unsigned char g = data[xy * 4 + 1];
        const unsigned char b = data[xy * 4 + 2];
        const unsigned char a = data[xy * 4 + 3];

        if (!(r == g && g == b)) {
            isGray = false;
        }
        if (a < 255) {
            hasAlpha = true;
        }
    }

    const int numChannels = (isGray ? 1 : 3) + (hasAlpha ? 1 : 0);

    return numChannels;
}

void compressImageData(const int width, const int height, const int channel, unsigned char* rawData, std::vector<unsigned char>& data)
{
    data.resize(width * height * channel);
    for (int xy = 0; xy < width * height; xy++) {
        if (channel > 0) {
            data[xy * channel + 0] = rawData[xy * 4 + 0];
        }
        if (channel == 2) {
            data[xy * channel + 1] = rawData[xy * 4 + 3];
        } else if (channel > 1) {
            data[xy * channel + 1] = rawData[xy * 4 + 1];
        }
        if (channel > 2) {
            data[xy * channel + 2] = rawData[xy * 4 + 2];
        }
        if (channel > 3) {
            data[xy * channel + 3] = rawData[xy * 4 + 3];
        }
    }
}

V2i pyramidLevelSize(const V2i& sizeBase, const int level)
{
    return V2i(V2f(sizeBase) * std::pow(2.0f, -float(level)));
}

std::string backendToString(const int ebsynthBackend)
{
    if (ebsynthBackend == EBSYNTH_BACKEND_CPU) {
        return "cpu";
    } else if (ebsynthBackend == EBSYNTH_BACKEND_CUDA) {
        return "cuda";
    } else if (ebsynthBackend == EBSYNTH_BACKEND_AUTO) {
        return "auto";
    }
    return "unknown";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("usage: %s [options]\n", argv[0]);
        printf("\n");
        printf("options:\n");
        printf("  -albedo <albedo.png>\n");
        printf("  -mask <mask.png>\n");
        printf("  -output <output.png>\n");
        printf("  -weight <value>\n");
        printf("  -uniformity <value>\n");
        printf("  -patchsize <size>\n");
        printf("  -pyramidlevels <number>\n");
        printf("  -searchvoteiters <number>\n");
        printf("  -patchmatchiters <number>\n");
        printf("  -stopthreshold <value>\n");
        printf("  -extrapass3x3\n");
        printf("  -backend [cpu|cuda]\n");
        printf("\n");
        return 1;
    }

    std::string albedoFileName = "albedo.png";
    std::string maskFileName   = "bmask0.png";
    std::string outputFileName;

    float albedoWeight = -1;
    float maskWeight   = -1;

    float uniformityWeight   = 3500;
    int   patchSize          = 5;
    int   numPyramidLevels   = -1;
    int   numSearchVoteIters = 6;
    int   numPatchMatchIters = 4;
    int   stopThreshold      = 5;
    int   extraPass3x3       = 0;
    int   backend            = ebsynthBackendAvailable(EBSYNTH_BACKEND_CUDA) ? EBSYNTH_BACKEND_CUDA : EBSYNTH_BACKEND_CPU;

    {
        std::vector<std::string> args(argc);
        for (int i = 0; i < argc; i++) {
            args[i] = argv[i];
        }

        bool fail = false;
        int  argi = 1;

        while (argi < argc && !fail) {
            float       weight;
            std::string backendName;

            if (tryToParseStringArg(args, &argi, "-albedo", &albedoFileName, &fail)) {
                albedoWeight = -1;
                argi++;
            } else if (tryToParseStringArg(args, &argi, "-mask", &maskFileName, &fail)) {
                maskWeight = -1;
                argi++;
            } else if (tryToParseStringArg(args, &argi, "-output", &outputFileName, &fail)) {
                argi++;
            } else if (tryToParseFloatArg(args, &argi, "-uniformity", &uniformityWeight, &fail)) {
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-patchsize", &patchSize, &fail)) {
                if (patchSize < 3) {
                    printf("error: patchsize is too small!\n");
                    return 1;
                }
                if (patchSize % 2 == 0) {
                    printf("error: patchsize must be an odd number!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-pyramidlevels", &numPyramidLevels, &fail)) {
                if (numPyramidLevels < 1) {
                    printf("error: bad argument for -pyramidlevels!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-searchvoteiters", &numSearchVoteIters, &fail)) {
                if (numSearchVoteIters < 0) {
                    printf("error: bad argument for -searchvoteiters!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-patchmatchiters", &numPatchMatchIters, &fail)) {
                if (numPatchMatchIters < 0) {
                    printf("error: bad argument for -patchmatchiters!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-stopthreshold", &stopThreshold, &fail)) {
                if (stopThreshold < 0) {
                    printf("error: bad argument for -stopthreshold!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseStringArg(args, &argi, "-backend", &backendName, &fail)) {
                if (backendName == "cpu") {
                    backend = EBSYNTH_BACKEND_CPU;
                } else if (backendName == "cuda") {
                    backend = EBSYNTH_BACKEND_CUDA;
                } else {
                    printf("error: unrecognized backend '%s'\n", backendName.c_str());
                    return 1;
                }

                if (!ebsynthBackendAvailable(backend)) {
                    printf("error: the %s backend is not available!\n", backendToString(backend).c_str());
                    return 1;
                }

                argi++;
            } else if (argi < args.size() && args[argi] == "-extrapass3x3") {
                extraPass3x3 = 1;
                argi++;
            } else {
                printf("error: unrecognized option '%s'\n", args[argi].c_str());
                fail = true;
            }
        }

        if (outputFileName.empty()) {
            size_t      lastIndex      = albedoFileName.find_last_of(".");
            std::string albedoBaseName = albedoFileName.substr(0, lastIndex);
            outputFileName             = albedoBaseName + "_inpainted.png";
        }

        if (fail) {
            return 1;
        }
    }

    int            albedoWidth            = 0;
    int            albedoHeight           = 0;
    unsigned char* albedoData             = tryLoad(albedoFileName, &albedoWidth, &albedoHeight);
    const int      numAlbedoChannelsTotal = evalNumChannels(albedoData, albedoWidth * albedoHeight);

    std::vector<unsigned char> albedoSource;
    compressImageData(albedoWidth, albedoHeight, numAlbedoChannelsTotal, albedoData, albedoSource);

    int            maskWidth            = 0;
    int            maskHeight           = 0;
    unsigned char* maskData             = tryLoad(maskFileName, &maskWidth, &maskHeight);
    const int      numMaskChannelsTotal = evalNumChannels(maskData, maskWidth * maskHeight);

    if (albedoWidth != maskWidth || albedoHeight != maskHeight) {
        printf("error: shape mismatch, source shape is %dx%dx%d, mask shape is %dx%dx%d\n", albedoWidth, albedoHeight, numAlbedoChannelsTotal, maskWidth, maskHeight, numMaskChannelsTotal);
        return 1;
    }
    if (numAlbedoChannelsTotal > EBSYNTH_MAX_STYLE_CHANNELS) {
        printf("error: too many style channels (%d), maximum number is %d\n", numAlbedoChannelsTotal, EBSYNTH_MAX_STYLE_CHANNELS);
        return 1;
    }
    if (numMaskChannelsTotal > EBSYNTH_MAX_GUIDE_CHANNELS) {
        printf("error: too many guide channels (%d), maximum number is %d\n", numMaskChannelsTotal, EBSYNTH_MAX_GUIDE_CHANNELS);
        return 1;
    }

    std::vector<unsigned char> maskSource;
    std::vector<unsigned char> maskTarget;
    compressImageData(maskWidth, maskHeight, numMaskChannelsTotal, maskData, maskTarget);
    for (auto& pixel : maskTarget) {
        maskSource.push_back(255 - pixel);
    }

    std::vector<float> albedoWeights(numAlbedoChannelsTotal);
    if (albedoWeight < 0) {
        albedoWeight = 1.0f;
    }
    for (int i = 0; i < numAlbedoChannelsTotal; i++) {
        albedoWeights[i] = albedoWeight / float(numAlbedoChannelsTotal);
    }

    std::vector<float> maskWeights(numMaskChannelsTotal);
    if (maskWeight < 0) {
        maskWeight = 1.0f;
    }
    for (int i = 0; i < numMaskChannelsTotal; i++) {
        maskWeights[i] = maskWeight / float(numMaskChannelsTotal);
    }

    int maxPyramidLevels = 0;
    for (int level = 32; level >= 0; level--) {
        if (min(pyramidLevelSize(V2i(albedoWidth, albedoHeight), level)) >= (2 * patchSize + 1)) {
            maxPyramidLevels = level + 1;
            break;
        }
    }

    if (numPyramidLevels == -1) {
        numPyramidLevels = maxPyramidLevels;
    }
    numPyramidLevels = std::min(numPyramidLevels, maxPyramidLevels);

    std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
    std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
    std::vector<int> stopThresholdPerLevel(numPyramidLevels);
    for (int i = 0; i < numPyramidLevels; i++) {
        numSearchVoteItersPerLevel[i] = numSearchVoteIters;
        numPatchMatchItersPerLevel[i] = numPatchMatchIters;
        stopThresholdPerLevel[i]      = stopThreshold;
    }

    std::vector<unsigned char> output(albedoWidth * albedoHeight * numAlbedoChannelsTotal);

    printf("uniformity: %.0f\n", uniformityWeight);
    printf("patchsize: %d\n", patchSize);
    printf("pyramidlevels: %d\n", numPyramidLevels);
    printf("searchvoteiters: %d\n", numSearchVoteIters);
    printf("patchmatchiters: %d\n", numPatchMatchIters);
    printf("stopthreshold: %d\n", stopThreshold);
    printf("extrapass3x3: %s\n", extraPass3x3 != 0 ? "yes" : "no");
    printf("backend: %s\n", backendToString(backend).c_str());

    ebsynthRun(backend,
               numAlbedoChannelsTotal,
               numMaskChannelsTotal,
               albedoWidth,
               albedoHeight,
               albedoSource.data(),
               maskSource.data(),
               maskWidth,
               maskHeight,
               maskTarget.data(),
               NULL,
               albedoWeights.data(),
               maskWeights.data(),
               uniformityWeight,
               patchSize,
               EBSYNTH_VOTEMODE_PLAIN,
               numPyramidLevels,
               numSearchVoteItersPerLevel.data(),
               numPatchMatchItersPerLevel.data(),
               stopThresholdPerLevel.data(),
               extraPass3x3,
               NULL,
               output.data());

    stbi_write_png(outputFileName.c_str(), albedoWidth, albedoHeight, numAlbedoChannelsTotal, output.data(), numAlbedoChannelsTotal * albedoWidth);

    printf("result was written to %s\n", outputFileName.c_str());

    stbi_image_free(albedoData);
    stbi_image_free(maskData);
    return 0;
}
