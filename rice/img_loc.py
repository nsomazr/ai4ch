from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exifread
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from datetime import datetime

def get_decimal_coordinates(info):
    """
    Convert GPS coordinates stored in EXIF to decimal degrees
    """
    try:
        for key in ['Latitude', 'Longitude']:
            if 'GPS' + key in info and 'GPS' + key + 'Ref' in info:
                e = info['GPS' + key]
                ref = info['GPS' + key + 'Ref']
                degrees = e[0][0] / e[0][1]
                minutes = e[1][0] / e[1][1]
                seconds = e[2][0] / e[2][1]
                
                if ref in ['S', 'W']:
                    degrees = -degrees
                    minutes = -minutes 
                    seconds = -seconds
                
                return degrees + (minutes / 60.0) + (seconds / 3600.0)
    except:
        return None

def get_location_from_coords(latitude, longitude):
    """
    Convert coordinates to location name using reverse geocoding
    """
    try:
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.reverse(f"{latitude}, {longitude}", language='en')
        
        if location:
            address = location.raw.get('address', {})
            return {
                'region': address.get('state', 'Unknown'),
                'district': address.get('county', address.get('city', 'Unknown')),
                'country': address.get('country', 'Unknown'),
                'full_address': location.address
            }
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass
    
    return None

def extract_image_location(image_file):
    """
    Extract location information from image EXIF data
    Works with both InMemoryUploadedFile and file paths
    """
    try:
        # Read EXIF data directly from the InMemoryUploadedFile
        if hasattr(image_file, 'read'):
            # If it's a file-like object (InMemoryUploadedFile)
            tags = exifread.process_file(image_file)
            # Reset file pointer for future reads
            image_file.seek(0)
        else:
            # If it's a file path
            with open(image_file, 'rb') as f:
                tags = exifread.process_file(f)
            
        # Get GPS tags
        gps_latitude = tags.get('GPS GPSLatitude')
        gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
        gps_longitude = tags.get('GPS GPSLongitude')
        gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
        
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = get_decimal_coordinates({
                'GPSLatitude': gps_latitude.values,
                'GPSLatitudeRef': gps_latitude_ref.values
            })
            lon = get_decimal_coordinates({
                'GPSLongitude': gps_longitude.values,
                'GPSLongitudeRef': gps_longitude_ref.values
            })
            
            if lat and lon:
                location_info = get_location_from_coords(lat, lon)
                if location_info:
                    location_info.update({
                        'latitude': lat,
                        'longitude': lon
                    })
                    return location_info
                    
    except Exception as e:
        print(f"Error extracting location: {str(e)}")
    
    return None