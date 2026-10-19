#import "tb_native_metal_renderer.h"

#import <AppKit/AppKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <os/lock.h>
#import <simd/simd.h>

#include <stdio.h>
#include <string.h>

@interface TBNativeMetalView : NSView
@end

@implementation TBNativeMetalView

- (CALayer *)makeBackingLayer {
    return [CAMetalLayer layer];
}

- (NSView *)hitTest:(NSPoint)point {
    (void)point;
    /* SDL's content view remains the sole input target. */
    return nil;
}

- (BOOL)acceptsFirstResponder {
    return NO;
}

@end

/* Cursor overlay: the receiver draws the sender-reported cursor type with the
 * matching native macOS cursor image (rendered once at 2x into a texture),
 * composited in the same pass as the video. */
#define TB_CURSOR_TYPE_COUNT 9

typedef struct {
    vector_float2 drawableSize;
    vector_float2 cursorOrigin;   /* top-left of the cursor image, drawable px */
    vector_float2 cursorExtent;   /* size of the cursor image, drawable px */
    uint32_t cursorVisible;
    uint32_t fullRange;
} TBNativeMetalUniforms;

static const char TBNativeMetalShader[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct RasterData {
    float4 position [[position]];
    float2 texCoord;
};

struct Uniforms {
    float2 drawableSize;
    float2 cursorOrigin;
    float2 cursorExtent;
    uint cursorVisible;
    uint fullRange;
};

vertex RasterData tbVideoVertex(uint vertexID [[vertex_id]]) {
    constexpr float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    constexpr float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    RasterData out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment float4 tbVideoFragment(RasterData in [[stage_in]],
                                texture2d<float, access::sample> luma [[texture(0)]],
                                texture2d<float, access::sample> chroma [[texture(1)]],
                                texture2d<float, access::sample> cursorImage [[texture(2)]],
                                constant Uniforms &uniforms [[buffer(0)]]) {
    constexpr sampler linearSampler(coord::normalized,
                                    address::clamp_to_edge,
                                    filter::linear);
    float y = luma.sample(linearSampler, in.texCoord).r;
    float2 uv = chroma.sample(linearSampler, in.texCoord).rg;

    if (uniforms.fullRange == 0) {
        y = max(0.0, (y - (16.0 / 255.0)) * (255.0 / 219.0));
        uv = (uv - 0.5) * (255.0 / 224.0);
    } else {
        uv -= 0.5;
    }

    float3 rgb;
    rgb.r = y + 1.5748 * uv.y;
    rgb.g = y - 0.1873 * uv.x - 0.4681 * uv.y;
    rgb.b = y + 1.8556 * uv.x;
    rgb = clamp(rgb, 0.0, 1.0);

    if (uniforms.cursorVisible != 0) {
        constexpr sampler cursorSampler(coord::normalized,
                                        address::clamp_to_edge,
                                        filter::linear);
        float2 p = (in.position.xy - uniforms.cursorOrigin) /
                   max(uniforms.cursorExtent, float2(1.0));
        if (all(p >= 0.0) && all(p <= 1.0)) {
            /* Premultiplied RGBA cursor image over the video. */
            float4 c = cursorImage.sample(cursorSampler, p);
            rgb = c.rgb + rgb * (1.0 - c.a);
        }
    }

    return float4(rgb, 1.0);
}
)METAL";

static BOOL tb_pixel_buffer_uses_display_p3(CVPixelBufferRef pixelBuffer) {
    if (!pixelBuffer) return NO;
    CFTypeRef primaries = CVBufferGetAttachment(
        pixelBuffer, kCVImageBufferColorPrimariesKey, NULL);
    return primaries && CFGetTypeID(primaries) == CFStringGetTypeID() &&
           CFEqual(primaries, kCVImageBufferColorPrimaries_P3_D65);
}

@interface TBNativeMetalRenderer : NSObject
- (instancetype)initRenderer;
- (void)setVisible:(BOOL)visible;
- (int)renderPixelBuffer:(CVPixelBufferRef)pixelBuffer
                 cursorX:(int)cursorX
                 cursorY:(int)cursorY
             cursorWidth:(int)cursorWidth
            cursorHeight:(int)cursorHeight
           cursorVisible:(BOOL)cursorVisible
              cursorType:(int)cursorType
             cursorLarge:(BOOL)cursorLarge
           rememberFrame:(BOOL)rememberFrame;
- (int)renderCursorX:(int)cursorX
             cursorY:(int)cursorY
         cursorWidth:(int)cursorWidth
        cursorHeight:(int)cursorHeight
       cursorVisible:(BOOL)cursorVisible
          cursorType:(int)cursorType
         cursorLarge:(BOOL)cursorLarge;
- (void)copyStats:(struct tb_native_metal_stats *)stats;
- (const char *)colorSpaceName;
- (void)waitUntilIdle;
@end

@implementation TBNativeMetalRenderer {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLRenderPipelineState> _pipeline;
    CVMetalTextureCacheRef _textureCache;
    TBNativeMetalView *_view;
    CAMetalLayer *_metalLayer;
    NSView *_sdlContentView;   /* SDL's view, swapped out while we own the window (TB_RECEIVER_METAL_OWN_WINDOW=1) */
    NSWindow *_ownedWindow;
    CGColorSpaceRef _layerColorSpace;
    dispatch_semaphore_t _inflightSemaphore;
    CVPixelBufferRef _latestFrame;
    os_unfair_lock _statsLock;
    struct tb_native_metal_stats _stats;
    BOOL _loggedFirstFrame;
    BOOL _displayP3;
    BOOL _cursorTexturesBuilt;
    id<MTLTexture> _cursorTextures[TB_CURSOR_TYPE_COUNT];
    NSSize _cursorPointSizes[TB_CURSOR_TYPE_COUNT];
    NSPoint _cursorHotspots[TB_CURSOR_TYPE_COUNT];
}

/* Sender cursor type ids (TBDisplaySenderService.swift): 0 arrow, 1 I-beam,
 * 2 pointing hand, 3 horizontal resize, 4 vertical resize, 5 closed/open hand,
 * 6 crosshair. 7/8 are reserved; they fall back to the arrow. */
static NSCursor *tb_native_cursor_for_type(int type) {
    switch (type) {
    case 1: return [NSCursor IBeamCursor];
    case 2: return [NSCursor pointingHandCursor];
    case 3: return [NSCursor resizeLeftRightCursor];
    case 4: return [NSCursor resizeUpDownCursor];
    case 5: return [NSCursor closedHandCursor];
    case 6: return [NSCursor crosshairCursor];
    case 7: return [NSCursor openHandCursor];
    default: return [NSCursor arrowCursor];
    }
}

/* Render each native cursor image once at 2x into an RGBA8 texture
 * (premultiplied alpha, top row first) and remember its point size and hot spot. */
- (void)buildCursorTexturesIfNeeded {
    if (_cursorTexturesBuilt || !_device) return;
    _cursorTexturesBuilt = YES;
    const CGFloat scale = 2.0;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    int built = 0;
    for (int type = 0; type < TB_CURSOR_TYPE_COUNT; type++) {
        NSCursor *cursor = tb_native_cursor_for_type(type);
        NSImage *image = cursor.image;
        const NSSize pointSize = image.size;
        if (!image || pointSize.width <= 0 || pointSize.height <= 0) continue;
        const size_t width = (size_t)ceil(pointSize.width * scale);
        const size_t height = (size_t)ceil(pointSize.height * scale);
        CGContextRef ctx = CGBitmapContextCreate(NULL, width, height, 8, width * 4, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        if (!ctx) continue;
        CGContextClearRect(ctx, CGRectMake(0, 0, width, height));
        CGContextScaleCTM(ctx, scale, scale); /* lets NSImage pick its @2x representation */
        NSGraphicsContext *gc = [NSGraphicsContext graphicsContextWithCGContext:ctx flipped:NO];
        [NSGraphicsContext saveGraphicsState];
        [NSGraphicsContext setCurrentContext:gc];
        [image drawInRect:NSMakeRect(0, 0, pointSize.width, pointSize.height)
                 fromRect:NSZeroRect
                operation:NSCompositingOperationSourceOver
                 fraction:1.0
           respectFlipped:YES
                    hints:@{ NSImageHintInterpolation: @(NSImageInterpolationHigh) }];
        [NSGraphicsContext restoreGraphicsState];

        MTLTextureDescriptor *desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:width
                                                              height:height
                                                           mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> texture = [_device newTextureWithDescriptor:desc];
        if (texture) {
            [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                       mipmapLevel:0
                         withBytes:CGBitmapContextGetData(ctx)
                       bytesPerRow:width * 4];
            _cursorTextures[type] = texture;
            _cursorPointSizes[type] = pointSize;
            _cursorHotspots[type] = cursor.hotSpot;
            built++;
        }
        CGContextRelease(ctx);
    }
    CGColorSpaceRelease(colorSpace);
    fprintf(stderr, "[metal-native] cursor textures built: %d/%d native shapes\n",
            built, TB_CURSOR_TYPE_COUNT);
}

- (instancetype)initRenderer {
    self = [super init];
    if (!self) return nil;

    _statsLock = OS_UNFAIR_LOCK_INIT;
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) return nil;
    _commandQueue = [_device newCommandQueue];
    if (!_commandQueue) return nil;

    NSError *libraryError = nil;
    NSString *shaderSource = [NSString stringWithUTF8String:TBNativeMetalShader];
    id<MTLLibrary> library = [_device newLibraryWithSource:shaderSource
                                                   options:nil
                                                     error:&libraryError];
    if (!library) {
        fprintf(stderr, "[metal-native] shader compile failed: %s\n",
                libraryError.localizedDescription.UTF8String ?: "unknown error");
        return nil;
    }

    MTLRenderPipelineDescriptor *descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.label = @"TargetBridge NV12 Pipeline";
    descriptor.vertexFunction = [library newFunctionWithName:@"tbVideoVertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@"tbVideoFragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    NSError *pipelineError = nil;
    _pipeline = [_device newRenderPipelineStateWithDescriptor:descriptor error:&pipelineError];
    if (!_pipeline) {
        fprintf(stderr, "[metal-native] pipeline creation failed: %s\n",
                pipelineError.localizedDescription.UTF8String ?: "unknown error");
        return nil;
    }

    CVReturn cacheStatus = CVMetalTextureCacheCreate(kCFAllocatorDefault,
                                                      NULL,
                                                      _device,
                                                      NULL,
                                                      &_textureCache);
    if (cacheStatus != kCVReturnSuccess || !_textureCache) {
        fprintf(stderr, "[metal-native] CVMetalTextureCacheCreate failed: %d\n",
                (int)cacheStatus);
        return nil;
    }

    _inflightSemaphore = dispatch_semaphore_create(3);
    fprintf(stderr, "[metal-native] device=%s\n", _device.name.UTF8String ?: "unknown");
    return self;
}

- (void)dealloc {
    [self setVisible:NO];
    if (_metalLayer) _metalLayer.colorspace = nil;
    if (_layerColorSpace) {
        CGColorSpaceRelease(_layerColorSpace);
        _layerColorSpace = NULL;
    }
    if (_textureCache) {
        CVMetalTextureCacheFlush(_textureCache, 0);
        CFRelease(_textureCache);
        _textureCache = NULL;
    }
}

- (NSWindow *)receiverWindow {
    NSWindow *best = nil;
    CGFloat bestArea = 0.0;
    for (NSWindow *window in NSApp.windows) {
        if (!window.isVisible) continue;
        const CGFloat area = window.frame.size.width * window.frame.size.height;
        if (area > bestArea) {
            bestArea = area;
            best = window;
        }
    }
    return best;
}

- (BOOL)attachViewIfNeeded {
    if (![NSThread isMainThread]) {
        fprintf(stderr, "[metal-native] UI attachment requested off main thread\n");
        return NO;
    }

    NSWindow *window = [self receiverWindow];
    if (!window) return NO;
    NSView *contentView = window.contentView;
    if (!contentView) return NO;
    if (_view.window == window && _view.superview == contentView && _metalLayer) return YES;
    if (_ownedWindow == window && window.contentView == _view && _metalLayer) return YES;

    /* SDL may replace its Cocoa window while entering its fullscreen Space.
     * Follow the largest visible receiver window instead of retaining the
     * initial 980x620 window forever. */
    if (_view) {
        if (_metalLayer) _metalLayer.colorspace = nil;
        if (_layerColorSpace) {
            CGColorSpaceRelease(_layerColorSpace);
            _layerColorSpace = NULL;
        }
        [_view removeFromSuperview];
        _view = nil;
        _metalLayer = nil;
    }

    _view = [[TBNativeMetalView alloc] initWithFrame:contentView.bounds];
    _view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _view.wantsLayer = YES;
    _view.hidden = YES;
    [contentView addSubview:_view positioned:NSWindowAbove relativeTo:nil];

    _metalLayer = (CAMetalLayer *)_view.layer;
    if (![_metalLayer isKindOfClass:CAMetalLayer.class]) return NO;
    _metalLayer.device = _device;
    _metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    CGColorSpaceRef srgb = tb_metal_colorspace_disabled() ? NULL : CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (srgb) {
        _metalLayer.colorspace = srgb;
        /* CAMetalLayer's CF-typed property does not declare ownership in the
         * Objective-C header. Retain our own reference until the layer is
         * detached so older macOS releases cannot observe a dangling space. */
        _layerColorSpace = srgb;
    }
    _displayP3 = NO;
    _metalLayer.framebufferOnly = YES;
    _metalLayer.opaque = YES;
    _metalLayer.presentsWithTransaction = NO;
    _metalLayer.allowsNextDrawableTimeout = YES;
    if ([_metalLayer respondsToSelector:@selector(setMaximumDrawableCount:)]) {
        _metalLayer.maximumDrawableCount = 3;
    }
    if ([_metalLayer respondsToSelector:@selector(setDisplaySyncEnabled:)]) {
        _metalLayer.displaySyncEnabled = tb_metal_display_sync_enabled();
    }
    fprintf(stderr,
            "[metal-native] attached window=%ld points=%.0fx%.0f scale=%.1f title=%s\n",
            (long)window.windowNumber,
            contentView.bounds.size.width,
            contentView.bounds.size.height,
            window.backingScaleFactor,
            window.title.UTF8String ?: "");
    return YES;
}

/* TB_RECEIVER_METAL_OWN_WINDOW=1: while video is visible, make the Metal view
 * the window's content view instead of a sublayer above SDL's OpenGL view, so
 * the window has a single opaque full-screen layer (direct-to-display
 * candidate). SDL's view is put back when the stream goes away. */
static BOOL tb_metal_own_window_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TB_RECEIVER_METAL_OWN_WINDOW");
        cached = (env && *env && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}

/* TB_RECEIVER_METAL_DISPLAY_SYNC=1 keeps CAMetalLayer vsync on (default off). */
static BOOL tb_metal_display_sync_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TB_RECEIVER_METAL_DISPLAY_SYNC");
        cached = (env && *env && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}

/* TB_RECEIVER_METAL_COLORSPACE=none leaves the layer without an explicit color
 * space: the WindowServer then has no color matching to do, which is one of
 * the conditions for direct-to-display presentation (no compositing pass). */
static BOOL tb_metal_colorspace_disabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("TB_RECEIVER_METAL_COLORSPACE");
        cached = (env && strcmp(env, "none") == 0) ? 1 : 0;
        if (cached) fprintf(stderr, "[metal-native] layer color space disabled (TB_RECEIVER_METAL_COLORSPACE=none)\n");
    }
    return cached == 1;
}

- (void)updateColorSpaceForPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    if (!_metalLayer) return;
    if (tb_metal_colorspace_disabled()) return;
    const BOOL displayP3 = tb_pixel_buffer_uses_display_p3(pixelBuffer);
    if (_metalLayer.colorspace && displayP3 == _displayP3) return;

    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(
        displayP3 ? kCGColorSpaceDisplayP3 : kCGColorSpaceSRGB);
    if (!colorSpace) return;
    CGColorSpaceRef previousColorSpace = _layerColorSpace;
    _layerColorSpace = colorSpace;
    _metalLayer.colorspace = colorSpace;
    if (previousColorSpace) CGColorSpaceRelease(previousColorSpace);
    _displayP3 = displayP3;
    fprintf(stderr, "[metal-native] output color space = %s\n",
            displayP3 ? "Display P3" : "sRGB");
}

- (void)updateDrawableSize {
    if (!_view || !_metalLayer) return;
    NSRect backingBounds = [_view convertRectToBacking:_view.bounds];
    const CGSize size = CGSizeMake(MAX(1.0, backingBounds.size.width),
                                   MAX(1.0, backingBounds.size.height));
    _metalLayer.contentsScale = _view.window.backingScaleFactor;
    if (!CGSizeEqualToSize(_metalLayer.drawableSize, size)) {
        _metalLayer.drawableSize = size;
        fprintf(stderr, "[metal-native] drawable %.0fx%.0f\n", size.width, size.height);
    }
}

- (void)setVisible:(BOOL)visible {
    @autoreleasepool {
        if (visible) {
            if ([self attachViewIfNeeded]) {
                if (tb_metal_own_window_enabled() && !_ownedWindow && _view.window) {
                    NSWindow *window = _view.window;
                    _sdlContentView = window.contentView;
                    [_view removeFromSuperview];
                    _view.frame = _sdlContentView.frame;
                    window.contentView = _view;
                    _ownedWindow = window;
                    fprintf(stderr, "[metal-native] Metal view now owns window %ld content view (SDL view parked)\n",
                            (long)window.windowNumber);
                }
                [self updateDrawableSize];
                _view.hidden = NO;
            }
            return;
        }

        if (_ownedWindow && _sdlContentView) {
            NSWindow *window = _ownedWindow;
            window.contentView = _sdlContentView;
            [_sdlContentView addSubview:_view positioned:NSWindowAbove relativeTo:nil];
            _view.frame = _sdlContentView.bounds;
            fprintf(stderr, "[metal-native] SDL view restored as window %ld content view\n",
                    (long)window.windowNumber);
            _ownedWindow = nil;
            _sdlContentView = nil;
        }
        if (_view) _view.hidden = YES;
        if (_latestFrame) {
            CVPixelBufferRelease(_latestFrame);
            _latestFrame = NULL;
        }
    }
}

- (void)recordDrop {
    os_unfair_lock_lock(&_statsLock);
    _stats.dropped_frames++;
    os_unfair_lock_unlock(&_statsLock);
}

- (int)renderPixelBuffer:(CVPixelBufferRef)pixelBuffer
                 cursorX:(int)cursorX
                 cursorY:(int)cursorY
             cursorWidth:(int)cursorWidth
            cursorHeight:(int)cursorHeight
           cursorVisible:(BOOL)cursorVisible
              cursorType:(int)cursorType
             cursorLarge:(BOOL)cursorLarge
           rememberFrame:(BOOL)rememberFrame {
    @autoreleasepool {
        if (!pixelBuffer || ![self attachViewIfNeeded]) return -1;
        [self updateDrawableSize];
        [self updateColorSpaceForPixelBuffer:pixelBuffer];
        _view.hidden = NO;

        if (rememberFrame && pixelBuffer != _latestFrame) {
            CVPixelBufferRetain(pixelBuffer);
            if (_latestFrame) CVPixelBufferRelease(_latestFrame);
            _latestFrame = pixelBuffer;
        }

        if (dispatch_semaphore_wait(_inflightSemaphore, DISPATCH_TIME_NOW) != 0) {
            [self recordDrop];
            return 0;
        }

        const size_t width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
        const size_t height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);
        const size_t chromaWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1);
        const size_t chromaHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1);

        CVMetalTextureRef lumaRef = NULL;
        CVMetalTextureRef chromaRef = NULL;
        CVReturn lumaStatus = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, _textureCache, pixelBuffer, NULL,
            MTLPixelFormatR8Unorm, width, height, 0, &lumaRef);
        CVReturn chromaStatus = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, _textureCache, pixelBuffer, NULL,
            MTLPixelFormatRG8Unorm, chromaWidth, chromaHeight, 1, &chromaRef);
        id<MTLTexture> lumaTexture = lumaRef ? CVMetalTextureGetTexture(lumaRef) : nil;
        id<MTLTexture> chromaTexture = chromaRef ? CVMetalTextureGetTexture(chromaRef) : nil;

        if (lumaStatus != kCVReturnSuccess || chromaStatus != kCVReturnSuccess ||
            !lumaTexture || !chromaTexture) {
            if (lumaRef) CFRelease(lumaRef);
            if (chromaRef) CFRelease(chromaRef);
            dispatch_semaphore_signal(_inflightSemaphore);
            fprintf(stderr, "[metal-native] CVMetalTexture creation failed: %d/%d\n",
                    (int)lumaStatus, (int)chromaStatus);
            return -1;
        }

        id<CAMetalDrawable> drawable = [_metalLayer nextDrawable];
        if (!drawable) {
            CFRelease(lumaRef);
            CFRelease(chromaRef);
            dispatch_semaphore_signal(_inflightSemaphore);
            [self recordDrop];
            return 0;
        }

        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

        TBNativeMetalUniforms uniforms;
        memset(&uniforms, 0, sizeof(uniforms));
        const CGSize drawableSize = _metalLayer.drawableSize;
        uniforms.drawableSize = (vector_float2){(float)drawableSize.width,
                                                (float)drawableSize.height};
        const float sourceWidth = (float)MAX(1, cursorWidth);
        const float sourceHeight = (float)MAX(1, cursorHeight);
        const float hotX = (float)cursorX * (float)drawableSize.width / sourceWidth;
        const float hotY = (float)cursorY * (float)drawableSize.height / sourceHeight;

        [self buildCursorTexturesIfNeeded];
        int type = (cursorType >= 0 && cursorType < TB_CURSOR_TYPE_COUNT) ? cursorType : 0;
        if (!_cursorTextures[type]) type = 0;
        id<MTLTexture> cursorTexture = _cursorTextures[type];
        /* Native size: one cursor point = one screen point (drawable px per
         * point from the view's backing scale). Large = 1.8x, like the
         * sender's accessibility option. */
        const float pointsToPx = (_view.bounds.size.width > 0.0)
            ? (float)(drawableSize.width / _view.bounds.size.width) : 2.0f;
        const float drawScale = pointsToPx * (cursorLarge ? 1.8f : 1.0f);
        uniforms.cursorOrigin = (vector_float2){
            hotX - (float)_cursorHotspots[type].x * drawScale,
            hotY - (float)_cursorHotspots[type].y * drawScale
        };
        uniforms.cursorExtent = (vector_float2){
            (float)_cursorPointSizes[type].width * drawScale,
            (float)_cursorPointSizes[type].height * drawScale
        };
        uniforms.cursorVisible = (cursorVisible && cursorTexture) ? 1u : 0u;
        const OSType format = CVPixelBufferGetPixelFormatType(pixelBuffer);
        uniforms.fullRange =
            format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ? 1u : 0u;

        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        commandBuffer.label = @"TargetBridge NV12 Frame";
        id<MTLRenderCommandEncoder> encoder =
            [commandBuffer renderCommandEncoderWithDescriptor:pass];
        [encoder setRenderPipelineState:_pipeline];
        [encoder setFragmentTexture:lumaTexture atIndex:0];
        [encoder setFragmentTexture:chromaTexture atIndex:1];
        if (cursorTexture) [encoder setFragmentTexture:cursorTexture atIndex:2];
        [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [encoder endEncoding];
        [commandBuffer presentDrawable:drawable];

        CVPixelBufferRetain(pixelBuffer);
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            double gpuMS = 0.0;
            if (completed.GPUEndTime > completed.GPUStartTime) {
                gpuMS = (completed.GPUEndTime - completed.GPUStartTime) * 1000.0;
            }
            CVPixelBufferRelease(pixelBuffer);
            CFRelease(lumaRef);
            CFRelease(chromaRef);
            /* Publish completion only after all retained frame resources are
             * released. Callers use this counter as a destruction barrier in
             * benchmarks and diagnostics, so incrementing it earlier exposed
             * a narrow use-after-release race at shutdown. */
            os_unfair_lock_lock(&self->_statsLock);
            self->_stats.completed_frames++;
            self->_stats.gpu_time_ms_total += gpuMS;
            if (gpuMS > self->_stats.gpu_time_ms_max) {
                self->_stats.gpu_time_ms_max = gpuMS;
            }
            os_unfair_lock_unlock(&self->_statsLock);
            dispatch_semaphore_signal(self->_inflightSemaphore);
        }];

        os_unfair_lock_lock(&_statsLock);
        _stats.submitted_frames++;
        os_unfair_lock_unlock(&_statsLock);
        [commandBuffer commit];

        if (!_loggedFirstFrame) {
            _loggedFirstFrame = YES;
            fprintf(stderr,
                    "[metal-native] first NV12 frame %zux%zu submitted without CPU upload\n",
                    width, height);
        }
        return 1;
    }
}

- (int)renderCursorX:(int)cursorX
             cursorY:(int)cursorY
         cursorWidth:(int)cursorWidth
        cursorHeight:(int)cursorHeight
       cursorVisible:(BOOL)cursorVisible
          cursorType:(int)cursorType
         cursorLarge:(BOOL)cursorLarge {
    if (!_latestFrame) return 0;
    return [self renderPixelBuffer:_latestFrame
                           cursorX:cursorX
                           cursorY:cursorY
                       cursorWidth:cursorWidth
                      cursorHeight:cursorHeight
                     cursorVisible:cursorVisible
                        cursorType:cursorType
                       cursorLarge:cursorLarge
                     rememberFrame:NO];
}

- (void)copyStats:(struct tb_native_metal_stats *)stats {
    if (!stats) return;
    os_unfair_lock_lock(&_statsLock);
    *stats = _stats;
    os_unfair_lock_unlock(&_statsLock);
}

- (const char *)colorSpaceName {
    return _displayP3 ? "Display P3" : "sRGB";
}

- (void)waitUntilIdle {
    /* Acquiring every in-flight slot waits for all command-buffer completion
     * handlers, then restores the semaphore for an orderly deallocation. */
    for (int slot = 0; slot < 3; slot++) {
        dispatch_semaphore_wait(_inflightSemaphore, DISPATCH_TIME_FOREVER);
    }
    for (int slot = 0; slot < 3; slot++) {
        dispatch_semaphore_signal(_inflightSemaphore);
    }
}

@end

void *tb_native_metal_create(void) {
    @autoreleasepool {
        TBNativeMetalRenderer *renderer = [[TBNativeMetalRenderer alloc] initRenderer];
        return renderer ? (__bridge_retained void *)renderer : NULL;
    }
}

void tb_native_metal_destroy(void *renderer) {
    if (!renderer) return;
    @autoreleasepool {
        TBNativeMetalRenderer *object = CFBridgingRelease(renderer);
        [object waitUntilIdle];
        [object setVisible:NO];
    }
}

void tb_native_metal_set_visible(void *renderer, int visible) {
    if (!renderer) return;
    @autoreleasepool {
        [(__bridge TBNativeMetalRenderer *)renderer setVisible:visible ? YES : NO];
    }
}

int tb_native_metal_render_nv12(void *renderer,
                                void *pixel_buffer,
                                int cursor_x,
                                int cursor_y,
                                int cursor_source_w,
                                int cursor_source_h,
                                int cursor_visible,
                                int cursor_type,
                                int cursor_large) {
    if (!renderer || !pixel_buffer) return -1;
    @autoreleasepool {
        return [(__bridge TBNativeMetalRenderer *)renderer
            renderPixelBuffer:(CVPixelBufferRef)pixel_buffer
                       cursorX:cursor_x
                       cursorY:cursor_y
                   cursorWidth:cursor_source_w
                  cursorHeight:cursor_source_h
                 cursorVisible:cursor_visible ? YES : NO
                    cursorType:cursor_type
                   cursorLarge:cursor_large ? YES : NO
                 rememberFrame:YES];
    }
}

int tb_native_metal_render_cursor(void *renderer,
                                  int cursor_x,
                                  int cursor_y,
                                  int cursor_source_w,
                                  int cursor_source_h,
                                  int cursor_visible,
                                  int cursor_type,
                                  int cursor_large) {
    if (!renderer) return -1;
    @autoreleasepool {
        return [(__bridge TBNativeMetalRenderer *)renderer
            renderCursorX:cursor_x
                 cursorY:cursor_y
             cursorWidth:cursor_source_w
            cursorHeight:cursor_source_h
           cursorVisible:cursor_visible ? YES : NO
              cursorType:cursor_type
             cursorLarge:cursor_large ? YES : NO];
    }
}

void tb_native_metal_get_stats(void *renderer,
                               struct tb_native_metal_stats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    if (!renderer) return;
    @autoreleasepool {
        [(__bridge TBNativeMetalRenderer *)renderer copyStats:stats];
    }
}

const char *tb_native_metal_pixel_buffer_color_space(void *pixel_buffer) {
    return tb_pixel_buffer_uses_display_p3((CVPixelBufferRef)pixel_buffer)
        ? "Display P3"
        : "sRGB";
}

const char *tb_native_metal_color_space_name(void *renderer) {
    if (!renderer) return "unavailable";
    @autoreleasepool {
        return [(__bridge TBNativeMetalRenderer *)renderer colorSpaceName];
    }
}
