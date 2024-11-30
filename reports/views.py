from django.shortcuts import render
from maize.models import MaizeData, MaizeDetectionResult, MaizePredictionResult
from rice.models import RiceData, RiceDetectionResult, RicePredictionResult
from cassava.models import CassavaData, CassavaDetectionResult, CassavaPredictionResult
from beans.models import BeansData, BeansDetectionResult, BeansPredictionResult
from django.shortcuts import render,get_object_or_404
# Create your views here.
from django.shortcuts import render
from django.core.paginator import Paginator
from rice.models import RiceData
from maize.models import MaizeData
from beans.models import BeansData
from cassava.models import CassavaData
from users.models import User  
import csv
from django.http import HttpResponseBadRequest
from django.http import HttpResponse

def get_file_type(file_path):
    video_extensions = ['mp4', 'avi', 'mov', 'mkv']
    image_extensions = ['jpg', 'jpeg', 'png', 'gif']
    extension = file_path.split('.')[-1].lower()
    if extension in video_extensions:
        return 'Video'
    elif extension in image_extensions:
        return 'Image'
    return 'Unknown'

def get_report_data():
    crop_models = [
        ("Rice", RiceData),
        ("Maize", MaizeData),
        ("Beans", BeansData),
        ("Cassava", CassavaData),
    ]
    data = []
    for crop_type, model in crop_models:
        for record in model.objects.all():
            user = record.uploaded_by
            data.append({
                "id": record.file_id,
                "email": user.email,
                "region": getattr(user, "region", "Unknown"),  # Ensure User has region and district
                "district": getattr(user, "district", "Unknown"),
                "file_type": get_file_type(str(record.file_path)),
                "crop_type": crop_type,
                "date": record.upload_date,
            })
    return data

def report_view(request):
    report_data = get_report_data()
    paginator = Paginator(report_data, 10)  # 10 items per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'backend/pages/reports.html', {"page_obj": page_obj})


def download_csv_report(request):
    report_data = get_report_data()  # Fetch the report data using the existing function
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="crop_report.csv"'

    writer = csv.writer(response)
    # Write the header
    writer.writerow(['ID', 'Email', 'Region', 'District', 'File Type', 'Crop Type', 'Date'])
    # Write the data rows
    for item in report_data:
        writer.writerow([
            item['id'],
            item['email'],
            item['region'],
            item['district'],
            item['file_type'],
            item['crop_type'],
            item['date'].strftime('%Y-%m-%d %H:%M:%S')  # Format the date
        ])
    
    return response


def result_detail(request, result_id,crop_type, result_type):
    """
    Dynamic view to handle prediction and detection details for Beans, Cassava, Rice, and Maize
    """
    print("Path: ", request.path.lower())
    try:
        if result_type == 'prediction':
            if str(crop_type).lower() == 'maize':
                result = get_object_or_404(MaizePredictionResult, result_id=result_id)
            elif str(crop_type).lower() =='beans':
                result = get_object_or_404(BeansPredictionResult, result_id=result_id)
            elif str(crop_type).lower() == 'cassava':
                result = get_object_or_404(CassavaPredictionResult, result_id=result_id)
            elif str(crop_type).lower() == 'rice':
                result = get_object_or_404(RicePredictionResult, result_id=result_id)
            else:
                raise ValueError("Unsupported crop type for prediction")

        elif result_type == 'detection':
            if str(crop_type).lower() == 'maize':
                result = get_object_or_404(MaizeDetectionResult, result_id=result_id)
            elif str(crop_type).lower() == 'beans':
                result = get_object_or_404(BeansDetectionResult, result_id=result_id)
            elif str(crop_type).lower() == 'cassava':
                result = get_object_or_404(CassavaDetectionResult, result_id=result_id)
            elif str(crop_type).lower() == 'rice':
                result = get_object_or_404(RiceDetectionResult, result_id=result_id)
            else:
                raise ValueError("Unsupported crop type for detection")

        else:
            raise ValueError("Invalid result type")

    except ValueError as e:
        # You might want to log this error or handle it more gracefully
        return HttpResponseBadRequest(str(e))

    return render(request, 'backend/pages/result_detail.html', {
        'result': result,
        'result_type': result_type
    })