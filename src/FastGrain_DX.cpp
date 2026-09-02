#include "FastGrain_DX.h"
#include "FastGrain_Shared.h"
#include <cstring>

#pragma comment(lib, "d3d12.lib")

static const unsigned kHeapSlots = 32;           /* 4 passes x 3 descriptors, with headroom */
static const unsigned kConstantBytes = 1024;     /* >= sizeof(FGParams), 256-aligned */

FGDX::~FGDX()
{
    if (mFence && mQueue) {
        WaitForGPU();
    }
    if (mConstants && mConstantsPtr) {
        mConstants->Unmap(0, nullptr);
        mConstantsPtr = nullptr;
    }
    if (mFenceEvent) {
        ::CloseHandle(mFenceEvent);
        mFenceEvent = nullptr;
    }
}

bool FGDX::Init(ID3D12Device* device, ID3D12CommandQueue* queue)
{
    if (!device || !queue) return false;
    mDevice = device;    /* ComPtr assignment AddRefs; AE keeps ownership */
    mQueue  = queue;

    if (FAILED(mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&mAllocator)))) return false;
    if (FAILED(mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, mAllocator.Get(), nullptr, IID_PPV_ARGS(&mList)))) return false;
    mList->Close();      /* Render() resets it */

    if (FAILED(mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFence)))) return false;
    mFenceEvent = ::CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!mFenceEvent) return false;

    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.NumDescriptors = kHeapSlots;
    hd.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(mDevice->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&mHeap)))) return false;
    mHandleSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_HEAP_PROPERTIES up = {};
    up.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = kConstantBytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(mDevice->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mConstants)))) return false;
    if (FAILED(mConstants->Map(0, nullptr, &mConstantsPtr))) return false;
    return true;
}

bool FGDX::LoadShader(int which, const void* cso, size_t csoBytes, const void* rootSig, size_t rootSigBytes)
{
    if (which < 0 || which >= kNumShaders || !cso || !rootSig) return false;
    if (!mRootSig) {
        if (FAILED(mDevice->CreateRootSignature(0, rootSig, rootSigBytes, IID_PPV_ARGS(&mRootSig)))) return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature    = mRootSig.Get();
    pd.CS.pShaderBytecode = cso;
    pd.CS.BytecodeLength  = csoBytes;
    return SUCCEEDED(mDevice->CreateComputePipelineState(&pd, IID_PPV_ARGS(&mPSO[which])));
}

void FGDX::MakeSRV(ID3D12Resource* res, uint64_t bytes, unsigned slot)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
    d.Format                  = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension           = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.FirstElement     = 0;
    d.Buffer.NumElements      = (UINT)(bytes / 4);
    d.Buffer.StructureByteStride = 0;
    d.Buffer.Flags            = D3D12_BUFFER_SRV_FLAG_RAW;
    D3D12_CPU_DESCRIPTOR_HANDLE h = mHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += (SIZE_T)slot * mHandleSize;
    mDevice->CreateShaderResourceView(res, &d, h);
}

void FGDX::MakeUAV(ID3D12Resource* res, uint64_t bytes, unsigned slot)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
    d.Format              = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension       = D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer.FirstElement = 0;
    d.Buffer.NumElements  = (UINT)(bytes / 4);
    d.Buffer.StructureByteStride = 0;
    d.Buffer.CounterOffsetInBytes = 0;
    d.Buffer.Flags        = D3D12_BUFFER_UAV_FLAG_RAW;
    D3D12_CPU_DESCRIPTOR_HANDLE h = mHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += (SIZE_T)slot * mHandleSize;
    mDevice->CreateUnorderedAccessView(res, nullptr, &d, h);
}

void FGDX::Transition(ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource   = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter  = after;
    mList->ResourceBarrier(1, &b);
}

void FGDX::Dispatch(int shader, ID3D12Resource* srv0, uint64_t srv0Bytes,
                    ID3D12Resource* srv1, uint64_t srv1Bytes,
                    ID3D12Resource* uav, uint64_t uavBytes,
                    unsigned groupsX, unsigned groupsY)
{
    /* three consecutive slots: SRV table (t0,t1) then UAV table (u0) */
    unsigned base = mNextSlot;
    mNextSlot += 3;
    MakeSRV(srv0, srv0Bytes, base + 0);
    MakeSRV(srv1, srv1Bytes, base + 1);
    MakeUAV(uav,  uavBytes,  base + 2);

    D3D12_GPU_DESCRIPTOR_HANDLE gpu = mHeap->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE srvTable = gpu; srvTable.ptr += (UINT64)(base + 0) * mHandleSize;
    D3D12_GPU_DESCRIPTOR_HANDLE uavTable = gpu; uavTable.ptr += (UINT64)(base + 2) * mHandleSize;

    mList->SetPipelineState(mPSO[shader].Get());
    mList->SetComputeRootConstantBufferView(0, mConstants->GetGPUVirtualAddress());
    mList->SetComputeRootDescriptorTable(1, srvTable);
    mList->SetComputeRootDescriptorTable(2, uavTable);
    mList->Dispatch(groupsX, groupsY, 1);
}

bool FGDX::WaitForGPU()
{
    const uint64_t v = ++mFenceValue;
    if (FAILED(mQueue->Signal(mFence.Get(), v))) return false;
    if (mFence->GetCompletedValue() < v) {
        if (FAILED(mFence->SetEventOnCompletion(v, mFenceEvent))) return false;
        ::WaitForSingleObject(mFenceEvent, INFINITE);
    }
    return true;
}

bool FGDX::Render(const FGParams& P,
                  ID3D12Resource* src,  uint64_t srcBytes,
                  ID3D12Resource* dst,  uint64_t dstBytes,
                  ID3D12Resource* latt, uint64_t lattBytes,
                  ID3D12Resource* temp, uint64_t tempBytes)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (int i = 0; i < kNumShaders; ++i) if (!mPSO[i]) return false;
    if (!src || !dst || !latt) return false;
    const bool blur = (P.blurRadius > 0) && temp != nullptr;

    std::memcpy(mConstantsPtr, &P, sizeof(FGParams));

    if (FAILED(mAllocator->Reset())) return false;
    if (FAILED(mList->Reset(mAllocator.Get(), nullptr))) return false;
    mNextSlot = 0;

    ID3D12DescriptorHeap* heaps[] = { mHeap.Get() };
    mList->SetDescriptorHeaps(1, heaps);
    mList->SetComputeRootSignature(mRootSig.Get());

    const unsigned gLx = (unsigned)((P.lattMaxW + 15) / 16);
    const unsigned gLy = (unsigned)((P.lattTotalH + 15) / 16);
    const unsigned gCx = (unsigned)((P.width + 15) / 16);
    const unsigned gCy = (unsigned)((P.height + 15) / 16);

    const D3D12_RESOURCE_STATES kSRV = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const D3D12_RESOURCE_STATES kUAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    /* buffers start in COMMON (implicit decay) and are promoted on first use */
    Dispatch(kGenLattice, src, srcBytes, src, srcBytes, latt, lattBytes, gLx, gLy);
    Transition(latt, kUAV, kSRV);

    if (blur) {
        Dispatch(kBlurH, latt, lattBytes, src, srcBytes, temp, tempBytes, gLx, gLy);
        Transition(temp, kUAV, kSRV);
        Transition(latt, kSRV, kUAV);
        Dispatch(kBlurV, temp, tempBytes, src, srcBytes, latt, lattBytes, gLx, gLy);
        Transition(latt, kUAV, kSRV);
    }

    Dispatch(kComposite, latt, lattBytes, src, srcBytes, dst, dstBytes, gCx, gCy);

    if (FAILED(mList->Close())) return false;
    ID3D12CommandList* lists[] = { mList.Get() };
    mQueue->ExecuteCommandLists(1, lists);
    return WaitForGPU();
}
