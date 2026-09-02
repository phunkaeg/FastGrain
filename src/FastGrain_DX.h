#pragma once
/*
    FastGrain_DX.h - minimal DirectX 12 compute helper.
    Shaders are embedded in the binary (generated FastGrain_Shaders.h), so no
    DirectX_Assets folder has to ship next to the .aex.
*/
#include <d3d12.h>
#include <wrl/client.h>
#include <mutex>
#include <cstdint>

struct FGParams;

class FGDX
{
public:
    enum Shader { kGenLattice = 0, kBlurH, kBlurV, kComposite, kNumShaders };

    FGDX() = default;
    ~FGDX();

    bool Init(ID3D12Device* device, ID3D12CommandQueue* queue);
    bool LoadShader(int which, const void* cso, size_t csoBytes, const void* rootSig, size_t rootSigBytes);

    /* Records and executes all passes, then waits for completion. Thread-safe. */
    bool Render(const FGParams& P,
                ID3D12Resource* src,  uint64_t srcBytes,
                ID3D12Resource* dst,  uint64_t dstBytes,
                ID3D12Resource* latt, uint64_t lattBytes,
                ID3D12Resource* temp, uint64_t tempBytes);

private:
    template <typename T> using ComPtr = Microsoft::WRL::ComPtr<T>;

    void MakeSRV(ID3D12Resource* res, uint64_t bytes, unsigned slot);
    void MakeUAV(ID3D12Resource* res, uint64_t bytes, unsigned slot);
    void Transition(ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
    void Dispatch(int shader, ID3D12Resource* srv0, uint64_t srv0Bytes,
                  ID3D12Resource* srv1, uint64_t srv1Bytes,
                  ID3D12Resource* uav, uint64_t uavBytes,
                  unsigned groupsX, unsigned groupsY);
    bool WaitForGPU();

    std::mutex                        mMutex;
    ComPtr<ID3D12Device>              mDevice;
    ComPtr<ID3D12CommandQueue>        mQueue;
    ComPtr<ID3D12CommandAllocator>    mAllocator;
    ComPtr<ID3D12GraphicsCommandList> mList;
    ComPtr<ID3D12Fence>               mFence;
    HANDLE                            mFenceEvent = nullptr;
    uint64_t                          mFenceValue = 0;

    ComPtr<ID3D12DescriptorHeap>      mHeap;
    unsigned                          mHandleSize = 0;
    unsigned                          mNextSlot = 0;

    ComPtr<ID3D12Resource>            mConstants;      /* upload heap, persistently mapped */
    void*                             mConstantsPtr = nullptr;

    ComPtr<ID3D12RootSignature>       mRootSig;
    ComPtr<ID3D12PipelineState>       mPSO[kNumShaders];
};
