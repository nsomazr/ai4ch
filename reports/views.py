from maize.models import MaizeData, MaizeDetectionResult, MaizePredictionResult
from rice.models import RiceData, RiceDetectionResult, RicePredictionResult
from cassava.models import CassavaData, CassavaDetectionResult, CassavaPredictionResult
from beans.models import BeansData, BeansDetectionResult, BeansPredictionResult
from django.shortcuts import render,get_object_or_404
# Create your views here.
from django.core.paginator import Paginator
from users.models import User  
import csv
from django.http import HttpResponseBadRequest
from django.http import HttpResponse
from django.db.models import Count
from django.db.models.functions import TruncDate, TruncWeek, TruncMonth
from datetime import datetime, timedelta
from django.utils import timezone
from itertools import chain
from django.db.models import Q

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
    # Map crop types to their data and detection models
    crop_models = [
        ("Rice", RiceData, RiceDetectionResult),
        ("Maize", MaizeData, MaizeDetectionResult),
        ("Beans", BeansData, BeansDetectionResult),
        ("Cassava", CassavaData, CassavaDetectionResult),
    ]
    
    data = []
    
    # Create detection results lookup dictionaries for each crop type
    detection_lookups = {}
    for crop_type, _, detection_model in crop_models:
        detection_lookups[crop_type] = {
            detection.result_id: detection.detection_results 
            for detection in detection_model.objects.all()
        }

    # Process each crop's records
    for crop_type, data_model, _ in crop_models:
        for record in data_model.objects.all():
            user = record.uploaded_by
            record_data = {
                "id": record.file_id,
                "email": user.email,
                "user_region": user.region,
                "user_district": user.district,
                "file_type": get_file_type(str(record.file_path)),
                "crop_type": crop_type,
                "image_region": record.region,
                "image_district": record.district,
                "date": record.upload_date,
            }
            
            # Look up detection results for this record
            detection_results = detection_lookups.get(crop_type, {}).get(record.file_id)
            record_data["detection_results"] = detection_results

            data.append(record_data)
            
    return data

def report_view(request):
    report_data = get_report_data()
    paginator = Paginator(report_data, 10)  # 10 items per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'backend/pages/reports.html', {"page_obj": page_obj})


def download_csv_report(request):
    report_data = get_report_data()
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="crop_report.csv"'

    writer = csv.writer(response)
    
    # Get all possible detection classes from all results
    detection_classes = set()
    for item in report_data:
        if item.get('detection_results'):
            detection_classes.update(item['detection_results'].keys())
    
    # Sort the detection classes for consistent column ordering
    detection_classes = sorted(detection_classes)
    
    # Write the header with detection classes as columns
    header = [
        'ID', 
        'Email', 
        'User Region', 
        'User District', 
        'File Type', 
        'Crop Type', 
        'Image Region', 
        'Image District', 
        'Date'
    ]
    # Add each detection class as a column
    header.extend(detection_classes)
    writer.writerow(header)

    # Write the data rows
    for item in report_data:
        detection_data = item.get('detection_results', {}) or {}
        
        # Build the basic row
        row = [
            item['id'],
            item['email'],
            item['user_region'],
            item['user_district'],
            item['file_type'],
            item['crop_type'],
            item['image_region'],
            item['image_district'],
            item['date'].strftime('%Y-%m-%d %H:%M:%S')
        ]
        
        # Add counts for each detection class
        for class_name in detection_classes:
            row.append(detection_data.get(class_name, 0))
            
        writer.writerow(row)
    
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
    
    

def analytics(request):
    # Get filter parameters
    time_filter = request.GET.get('time_filter', 'all')
    start_date = None
    
    # Calculate date range based on filter
    if time_filter == 'day':
        start_date = timezone.now().date()
    elif time_filter == 'week':
        start_date = timezone.now().date() - timedelta(days=7)
    elif time_filter == 'month':
        start_date = timezone.now().date() - timedelta(days=30)
    
    # Base querysets
    rice_qs = RiceData.objects.all()
    maize_qs = MaizeData.objects.all()
    beans_qs = BeansData.objects.all()
    cassava_qs = CassavaData.objects.all()
    
    # Apply date filter if specified
    if start_date:
        rice_qs = rice_qs.filter(upload_date__gte=start_date)
        maize_qs = maize_qs.filter(upload_date__gte=start_date)
        beans_qs = beans_qs.filter(upload_date__gte=start_date)
        cassava_qs = cassava_qs.filter(upload_date__gte=start_date)
    
    # Users per region data
    users_per_region = User.objects.values('region').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Crop type distribution
    crop_distribution = [
        {'name': 'Rice', 'count': rice_qs.count()},
        {'name': 'Maize', 'count': maize_qs.count()},
        {'name': 'Beans', 'count': beans_qs.count()},
        {'name': 'Cassava', 'count': cassava_qs.count()}
    ]
    
    # Get prediction results counts
    prediction_counts = {
        'Rice': RicePredictionResult.objects.filter(
            result_id__in=rice_qs.values('file_id')
        ).count(),
        'Maize': MaizePredictionResult.objects.filter(
            result_id__in=maize_qs.values('file_id')
        ).count(),
        'Beans': BeansPredictionResult.objects.filter(
            result_id__in=beans_qs.values('file_id')
        ).count(),
        'Cassava': CassavaPredictionResult.objects.filter(
            result_id__in=cassava_qs.values('file_id')
        ).count()
    }
    
    # Get file type distribution
    file_types = []
    for qs in [rice_qs, maize_qs, beans_qs, cassava_qs]:
        for record in qs:
            file_types.append(get_file_type(str(record.file_path)))
    
    file_type_distribution = {
        file_type: file_types.count(file_type)
        for file_type in set(file_types)
    }
    
    
    context = {
        'users_per_region': list(users_per_region),
        'crop_distribution': crop_distribution,
        'prediction_counts': prediction_counts,
        'file_type_distribution': file_type_distribution,
        'selected_filter': time_filter
    }
    
    return render(request, 'backend/pages/analytics.html', context)
        