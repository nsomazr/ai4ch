# Generated by Django 5.0 on 2024-11-22 01:04

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0007_platformuser_role_alter_platformuser_district_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="platformuser",
            name="role",
            field=models.CharField(
                choices=[("admin", "Admin"), ("manager", "Manager")],
                default="normal",
                max_length=20,
            ),
        ),
    ]