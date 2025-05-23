// Include necessary headers
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"

// Function to process an image
void UClosetScanComponent::ProcessClothingImage(const FString& ImagePath)
{
    // Initialize HTTP request
    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
    
    // Set the URL and verb
    HttpRequest->SetURL(TEXT("http://localhost:8000/process_image"));
    HttpRequest->SetVerb(TEXT("POST"));
    
    // Create multipart form data
    FString Boundary = TEXT("---------------------------" + FString::FromInt(FMath::Rand()));
    HttpRequest->SetHeader(TEXT("Content-Type"), TEXT("multipart/form-data; boundary=" + Boundary));
    
    // Read the image file
    TArray<uint8> ImageData;
    if (FFileHelper::LoadFileToArray(ImageData, *ImagePath))
    {
        // Build the form data
        TArray<uint8> RequestContent;
        FString BeginBoundary = TEXT("\r\n--" + Boundary + "\r\n");
        FString EndBoundary = TEXT("\r\n--" + Boundary + "--\r\n");
        
        // Add the file part
        FString FileHeader = BeginBoundary +
            TEXT("Content-Disposition: form-data; name=\"file\"; filename=\"") + FPaths::GetCleanFilename(ImagePath) + TEXT("\"\r\n") +
            TEXT("Content-Type: image/jpeg\r\n\r\n");
        
        // Convert string to bytes and append
        FTCHARToUTF8 FileHeaderUtf8(*FileHeader);
        RequestContent.Append((uint8*)FileHeaderUtf8.Get(), FileHeaderUtf8.Length());
        
        // Append image data
        RequestContent.Append(ImageData);
        
        // Add make_seamless parameter
        FString ParamHeader = BeginBoundary +
            TEXT("Content-Disposition: form-data; name=\"make_seamless\"\r\n\r\n") +
            TEXT("false");
        
        // Convert string to bytes and append
        FTCHARToUTF8 ParamHeaderUtf8(*ParamHeader);
        RequestContent.Append((uint8*)ParamHeaderUtf8.Get(), ParamHeaderUtf8.Length());
        
        // Add end boundary
        FTCHARToUTF8 EndBoundaryUtf8(*EndBoundary);
        RequestContent.Append((uint8*)EndBoundaryUtf8.Get(), EndBoundaryUtf8.Length());
        
        // Set content and send request
        HttpRequest->SetContent(RequestContent);
        
        // Set up response callback
        HttpRequest->OnProcessRequestComplete().BindUObject(this, &UClosetScanComponent::OnImageProcessed);
        
        // Send the request
        HttpRequest->ProcessRequest();
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to load image file: %s"), *ImagePath);
    }
}

// Callback for when the request completes
void UClosetScanComponent::OnImageProcessed(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSuccess)
{
    if (bSuccess && Response.IsValid())
    {
        if (Response->GetResponseCode() == 200)
        {
            // Get the response content
            const TArray<uint8>& ResponseContent = Response->GetContent();
            
            // Save the processed image
            FString OutputPath = FPaths::ProjectSavedDir() + TEXT("ProcessedTexture.png");
            if (FFileHelper::SaveArrayToFile(ResponseContent, *OutputPath))
            {
                UE_LOG(LogTemp, Display, TEXT("Processed texture saved to: %s"), *OutputPath);
                
                // Load the texture
                UTexture2D* ProcessedTexture = LoadTextureFromFile(OutputPath);
                if (ProcessedTexture)
                {
                    // Use the texture (e.g., apply to a material)
                    ApplyTextureToMaterial(ProcessedTexture);
                }
            }
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("API request failed with code: %d"), Response->GetResponseCode());
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("API request failed"));
    }
}